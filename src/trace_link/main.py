import os,sys,json
import torch
import numpy as np
from tqdm import tqdm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.model.calculate_code_vectors import get_pt_file_name
from src.utils.utils import load_config, get_trace_link_result_file_name, get_requirements_processed_file_name
from src.JavaCodeAnalyzer.tree_sitter_java_analyzer import analyze_directory
from src.model.calculate_code_vectors import process_analysis_files
from src.LLMapi.LLM_tset import check_requirement_code_relation
from src.model.encoder_factory import EncoderFactory
from src.trace_link.calculate import calculate_recall

CONFIG = load_config()
encoder = None
data = None
use_llm_processing = CONFIG["requirement_processing"]["use_llm_processing"]
encode_model_name = CONFIG.get("encode_model_name", "jina-code")
top_k_list = CONFIG.get("top_k", [5])
if isinstance(top_k_list, int):
    top_k_list = [top_k_list]
top_k_list = sorted(set(top_k_list))

def get_data():
    # 根据模型名称生成pt文件名
    pt_file_name = get_pt_file_name()
    pt_file_path = os.path.join('data', CONFIG['repo'], pt_file_name)
    if not os.path.exists(pt_file_path):
        print(f"文件不存在: {pt_file_path}")
        exit(1)
    data_ = torch.load(pt_file_path)
    return data_

def get_encoder():
    # 创建编码器并计算需求向量
    print(f"正在使用编码器: {encode_model_name}")
    encoder_ = EncoderFactory.create_encoder(encode_model_name)
    return encoder_

def load_requirements():
    """加载处理后的需求数据"""
    req_file = os.path.join('data', CONFIG['repo'], get_requirements_processed_file_name())
    if os.path.exists(req_file):
        with open(req_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def filter_req_by_type(requirements):
    """过滤需求类型"""
    req_type_list = CONFIG['req_type']
    return [req for req in requirements if req.get('type', '') in req_type_list]

def get_source_file_path(file_path):
    """获取源代码文件的完整路径"""
    # 从 change_files 中的相对路径转换为绝对路径
    src_dir = os.path.join('data', CONFIG['repo'], 'origin_src')
    return os.path.join(src_dir, file_path)


def check_requirement_code_relation_llm(req, file_path, code_snippet=""):
    """使用LLM判断需求和代码文件是否语义相关
    Args:
        req: 需求对象
        file_path: 代码文件路径
        code_snippet: 代码片段，默认为空字符串，用于直接传递代码内容
    Returns:
        包含关系信息的字典
    """
    # 获取完整文件路径
    full_path = get_source_file_path(file_path)
    if not os.path.exists(full_path):
        print(f"文件不存在: {full_path}")
        exit(1)
    # 读取代码文件内容
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            code_content = f.read()
    except Exception as e:
        print(f"读取文件 {full_path} 时出错: {e}")
        exit(1)
    
    # 构建需求文本
    req_title = req.get('title', '')
    req_description = req.get('description', '')
    requirement_text = f"{req_title}\n{req_description}"
    
    # 调用LLM API判断关系
    try:
        result_json = check_requirement_code_relation(requirement_text, code_content)
        # 解析JSON结果
        result = json.loads(result_json)
        return result
    except Exception as e:
        return {
            'related': False,
            'reason': f'LLM调用失败: {str(e)}',
            'confidence': 0.0
        }


def process_files_with_encoder(req, change_files):
    # 获取配置的代码片段类型
    code_snippet_types = CONFIG.get("code_snippet", ["default"])
    
    links_all = {}

    # 构建需求文本
    requirement_text = req.get('search_query', '')
    req_title = req.get('title', '')
    if CONFIG["requirement_processing"]["prefix_title"]:
        requirement_text = f"{req_title}\n{requirement_text}"
    req_embedding = encoder.encode_query([requirement_text])[0]
    req_embedding = torch.tensor(req_embedding)

    embeddings = data['embeddings']
    snippet_types = data.get('snippet_types', [])
    device = req_embedding.device
    embeddings = embeddings.to(device)

    # 归一化（如果当初没做）
    req_embedding = torch.nn.functional.normalize(req_embedding, dim=0)
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)

    # 一次性算相似度
    similarities = torch.matmul(
        req_embedding.unsqueeze(0),
        embeddings.T
    ).squeeze(0)

    # 预排序所有候选（一次性计算）
    sorted_indices = torch.argsort(similarities, descending=True)

    file_paths = data['file_paths']
    class_names = data['class_names']
    method_names = data['method_names']
    original_codes = data['original_code']
    allowed_snippets = set(code_snippet_types)
    unique_file_only = CONFIG["unique_file_only"]

    def get_snippet_suffix(st):
        return st.split("_", 1)[1] if "_" in st else st

    links_all = {}
    seen_file_paths = set()
    current_pointer = 0
    links = []

    for top_k in top_k_list:
        k = min(top_k, embeddings.shape[0])

        while len(links) < k and current_pointer < len(sorted_indices):
            idx = sorted_indices[current_pointer].item()
            current_pointer += 1

            file_path = file_paths[idx]
            snippet_type = snippet_types[idx] if idx < len(snippet_types) else "unknown"
            suffix = get_snippet_suffix(snippet_type)

            if suffix not in allowed_snippets:
                continue

            if (not unique_file_only) or (file_path not in seen_file_paths):
                seen_file_paths.add(file_path)
                links.append({
                    'file_path': file_path,
                    'class_name': class_names[idx],
                    'method_name': method_names[idx],
                    'similarity': similarities[idx].item(),
                    'original_code': original_codes[idx],
                    'snippet_type': snippet_type
                })

        links_all[top_k] = list(links)

    return links_all

def trace_links():
    """实现需求到代码的追踪链接"""
    global encoder, data ,CONFIG
    CONFIG = load_config()
    
    # 初始化编码器和数据（如果尚未初始化）
    if encoder is None:
        encoder = get_encoder()
    if data is None:
        data = get_data()
    
    print("开始实现需求到代码的追踪链接...")
    
    # 加载需求数据
    requirements = load_requirements()
    print(f"加载了 {len(requirements)} 个需求")
    
    if CONFIG['requirement_processing']['filter_invalid_issues']:
    # 过滤无效需求
        print(f"开始过滤需求，保留类型: {CONFIG['req_type']}")
        requirements = filter_req_by_type(requirements)
        print(f"过滤后 {len(requirements)} 个需求")

    results = []
    
    # 整体统计（按topk分别统计）
    overall_stats = {}
    for top_k in top_k_list:
        overall_stats[top_k] = {
            'total_requirements': len(requirements),
            'requirements_with_change_files': 0,
            'requirements_with_at_least_one_hit': 0,
            'total_change_files': 0,
            'total_predicted_files': 0,
            'total_hit_files': 0,
            'total_fp_files': 0,
            'top_k': top_k,
            'average_recall': 0.0,
            'average_precision': 0.0,
            'average_f1': 0.0,
            'total_recall_sum': 0.0,
            'total_precision_sum': 0.0,
            'total_f1_sum': 0.0
        }
    
    for req in tqdm(requirements, desc="已完成追踪连接需求："):
        req_id = req.get('req_id')
        req_title = req.get('title')
        change_files = req.get('change_files', [])
        has_change_files = len(change_files) > 0
        
        # 处理文件并计算相似度
        links_all = process_files_with_encoder(req, change_files)
        
        # 为每个topk处理LLM判断和召回率
        req_recalls = {}
        all_links = {}
        
        # 先收集所有topk的链接
        for top_k, links in links_all.items():
            if CONFIG['trace_link']['use_llm']:
                for link in links:
                    file_path = link['file_path']
                    # 使用LLM判断关系
                    llm_result = check_requirement_code_relation_llm(req, file_path)
                    link['llm_result'] = llm_result
                #根据LLM结果排序，related为true且confidence高的排在前面
                links.sort(key=lambda x: (not x['llm_result']['related'], -x['llm_result']['confidence']))
            all_links[top_k] = links
        
        # 计算每个topk的召回率
        if has_change_files:
            for top_k, links in all_links.items():
                recall_info = calculate_recall(links, change_files)
                req_recalls[top_k] = recall_info
                
                # 更新整体统计
                stats = overall_stats[top_k]
                stats['requirements_with_change_files'] += 1
                stats['total_change_files'] += recall_info['total_change_files']
                stats['total_predicted_files'] += recall_info['predicted_count']
                stats['total_hit_files'] += recall_info['hit_count']
                stats['total_fp_files'] += recall_info['fp_count']
                if recall_info['hit_count'] > 0:
                    stats['requirements_with_at_least_one_hit'] += 1
                stats['total_recall_sum'] += recall_info['recall']
                stats['total_precision_sum'] += recall_info['precision']
                stats['total_f1_sum'] += recall_info['f1']
        
        # 选择最大topk的链接作为主链接
        max_topk = max(top_k_list)
        main_links = all_links[max_topk]
        
        # 添加到结果
        result_item = {
            'req_id': req_id,
            'req_title': req_title,
            'req_description': req.get('description', ''),
            'req_search_query': req.get('search_query', ''),
            'req_url': req.get('url', ''),
            'req_type': req.get('type', 'default'),
            'links': main_links[:5],  # 最多显示前5个链接
            'change_files': change_files,
            'recall': req_recalls if has_change_files else None
        }
        
        results.append(result_item)
    
    # 计算整体统计
    for top_k, stats in overall_stats.items():
        if stats['total_change_files'] > 0:
            stats['overall_recall'] = stats['total_hit_files'] / stats['total_change_files']
        else:
            stats['overall_recall'] = 0.0

        if stats['total_predicted_files'] > 0:
            stats['overall_precision'] = stats['total_hit_files'] / stats['total_predicted_files']
        else:
            stats['overall_precision'] = 0.0

        if (stats['overall_recall'] + stats['overall_precision']) > 0:
            stats['overall_f1'] = 2 * stats['overall_recall'] * stats['overall_precision'] / (stats['overall_recall'] + stats['overall_precision'])
        else:
            stats['overall_f1'] = 0.0

        if stats['requirements_with_change_files'] > 0:
            stats['average_recall'] = stats['total_recall_sum'] / stats['requirements_with_change_files']
            stats['average_precision'] = stats['total_precision_sum'] / stats['requirements_with_change_files']
            stats['average_f1'] = stats['total_f1_sum'] / stats['requirements_with_change_files']
        else:
            stats['average_recall'] = 0.0
            stats['average_precision'] = 0.0
            stats['average_f1'] = 0.0

        del stats['total_recall_sum']
        del stats['total_precision_sum']
        del stats['total_f1_sum']

    # 准备最终输出
    final_output = {
        'results': results,
        'statistics': overall_stats
    }

    # 保存完整结果
    output_file = os.path.join('data', CONFIG['repo'],'trace_link_results', get_trace_link_result_file_name())
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False, separators=(',', ': '))

    print(f"\n追踪链接结果已保存到: {output_file}")

    # 保存统计结果（仅statistics + config）
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    llm_provider_name = CONFIG.get('LLMProvider')
    llm_config = CONFIG.get(llm_provider_name, {}).copy() if llm_provider_name else {}
    llm_config.pop('API_Key', None)

    tlr_config = {
        'encode_model_name': CONFIG.get('encode_model_name'),
        'code_snippet': CONFIG.get('code_snippet'),
        'top_k': CONFIG.get('top_k'),
        'unique_file_only': CONFIG.get('unique_file_only'),
        'requirement_processing': CONFIG.get('requirement_processing'),
        'trace_link': CONFIG.get('trace_link'),
        'code_embedding': CONFIG.get('code_embedding'),
        'repo': CONFIG.get('repo'),
        'LLMProvider': llm_provider_name,
        'LLMtemperature': CONFIG.get('LLMtemperature'),
        llm_provider_name: llm_config
    }

    stats_output = {
        'statistics': overall_stats,
        'config': tlr_config
    }

    stats_dir = os.path.join('data', CONFIG['repo'], 'statistics_results')
    os.makedirs(stats_dir, exist_ok=True)
    stats_file = os.path.join(stats_dir, f"{timestamp}.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_output, f, indent=2, ensure_ascii=False, separators=(',', ': '))

    print(f"统计结果已保存到: {stats_file}")

    # 打印结果摘要
    for top_k, stats in overall_stats.items():
        print(f"\n" + "=" * 60)
        print(f"Top {top_k} 整体统计:")
        print("=" * 60)
        print(f"  总需求数: {stats['total_requirements']}")
        print(f"  有变更文件的需求数: {stats['requirements_with_change_files']}")
        print(f"  至少命中一个文件的需求数: {stats['requirements_with_at_least_one_hit']}")
        print(f"  总变更文件数 (Actual): {stats['total_change_files']}")
        print(f"  预测文件数 (Predicted): {stats['total_predicted_files']}")
        print(f"  命中文件数 (TP): {stats['total_hit_files']}")
        print(f"  误报文件数 (FP): {stats['total_fp_files']}")
        print(f"  整体召回率 (Recall): {stats.get('overall_recall', 0):.4f}")
        print(f"  整体准确率 (Precision): {stats.get('overall_precision', 0):.4f}")
        print(f"  整体F1分数 (F1): {stats.get('overall_f1', 0):.4f}")
        print(f"  平均召回率: {stats.get('average_recall', 0):.4f}")
        print(f"  平均准确率: {stats.get('average_precision', 0):.4f}")
        print(f"  平均F1分数: {stats.get('average_f1', 0):.4f}")
        print("=" * 60)
    
    # 打印前几个需求的结果
    for result in results[:3]:  # 只显示前3个需求
        print(f"\n需求 {result['req_id']}: {result['req_title']}")
        if result['change_files']:
            if result.get('recall'):
                for top_k, recall_info in result['recall'].items():
                    print(f"  Top {top_k}: 变更文件: {recall_info['total_change_files']}, 预测: {recall_info['predicted_count']}, 命中: {recall_info['hit_count']}, FP: {recall_info['fp_count']}, Recall: {recall_info['recall']:.4f}, Precision: {recall_info['precision']:.4f}, F1: {recall_info['f1']:.4f}")
        print("  最相似的链接:")
        for link in result['links'][:3]:
            print(f"    - 文件: {link['file_path']}, 类: {link['class_name']}, 相似度: {link['similarity']:.4f}")


def get_data():
    # 根据模型名称生成pt文件名
    pt_file_name = get_pt_file_name()
    pt_file_path = os.path.join('data', CONFIG['repo'], pt_file_name)
    if not os.path.exists(pt_file_path):
        print(f"文件不存在: {pt_file_path}")
        exit(1)
    data_ = torch.load(pt_file_path)
    return data_


if __name__ == "__main__":
    test_directory = f"data\\{CONFIG['repo']}\\origin_src"
    analyze_directory(test_directory)
    process_analysis_files(test_directory)
    trace_links()


