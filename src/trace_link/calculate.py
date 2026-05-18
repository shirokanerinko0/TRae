import os,sys,json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.utils import load_config
CONFIG = load_config()
def calculate_ap(links, change_files, max_rank=1000):
    """
    计算单条需求的 Average Precision (AP)

    Args:
        links (list): 预测的链接列表，按相似度排序，每个元素包含 'file_path' 字段
        change_files (list): 实际变更的文件列表
        max_rank (int): 超过此排名的正确答案精度加权视为0

    Returns:
        float: AP值
    """
    actual_file_set = set()
    for cf in change_files:
        if isinstance(cf, str):
            file_path = cf
        elif isinstance(cf, dict) and 'file_path' in cf:
            file_path = cf['file_path']
        else:
            continue
        normalized_path = os.path.normpath(file_path)
        actual_file_set.add(normalized_path)

    exclude_files = set()
    if CONFIG.get('filter_req_exclude_dirs', False):
        for file in actual_file_set.copy():
            if any(exclude_dir in file for exclude_dir in CONFIG.get('exclude_dirs', [])):
                exclude_files.add(file)
        actual_file_set = actual_file_set - exclude_files

    num_relevant = len(actual_file_set)
    if num_relevant == 0:
        return 0.0

    hits = 0
    sum_precision = 0.0

    for k, link in enumerate(links, 1):
        if k > max_rank:
            break

        normalized_path = os.path.normpath(link['file_path'])
        if normalized_path in actual_file_set:
            hits += 1
            precision_at_k = hits / k
            sum_precision += precision_at_k

    ap = sum_precision / num_relevant
    return ap


def calculate_recall(links, change_files):
    """
    计算召回率和准确率

    Args:
        links (list): 预测的链接列表，每个元素包含 'file_path' 字段
        change_files (list): 实际变更的文件列表，每个元素可以是字符串或包含 'file_path' 字段的字典

    Returns:
        dict: 包含召回统计信息的字典
            - total_change_files: 变更文件总数 (actual count)
            - predicted_count: 预测的文件总数 (|Predicted|)
            - hit_count: 命中的变更文件数 (TP)
            - miss_count: 未命中的变更文件数 (FN)
            - fp_count: 误报数 (FP) = 预测中未命中的文件数
            - recall: 召回率 (TP / (TP + FN)) = hit_count / total_change_files
            - precision: 准确率 (TP / (TP + FP)) = hit_count / predicted_count
            - f1: F1分数 = 2 * precision * recall / (precision + recall)
            - hit_files: 命中的文件列表
            - missed_files: 未命中的文件列表
            - fp_files: 误报的文件列表（预测了但实际未变更）
    """
    actual_file_set = set()
    for cf in change_files:
        if isinstance(cf, str):
            file_path = cf
        elif isinstance(cf, dict) and 'file_path' in cf:
            file_path = cf['file_path']
        else:
            continue
        normalized_path = os.path.normpath(file_path)
        actual_file_set.add(normalized_path)

    exclude_files = set()
    if CONFIG.get('filter_req_exclude_dirs', False):
        for file in actual_file_set.copy():
            if any(exclude_dir in file for exclude_dir in CONFIG.get('exclude_dirs', [])):
                exclude_files.add(file)
        actual_file_set = actual_file_set - exclude_files

    predicted_file_set = set()
    for link in links:
        if 'file_path' in link:
            normalized_path = os.path.normpath(link['file_path'])
            predicted_file_set.add(normalized_path)

    hit_files = actual_file_set & predicted_file_set
    missed_files = actual_file_set - predicted_file_set
    fp_files = predicted_file_set - actual_file_set

    total_change_files = len(actual_file_set)
    predicted_count = len(predicted_file_set)
    hit_count = len(hit_files)
    miss_count = len(missed_files)
    fp_count = len(fp_files)

    recall = hit_count / total_change_files if total_change_files > 0 else 0.0
    precision = hit_count / predicted_count if predicted_count > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'total_change_files': total_change_files,
        'predicted_count': predicted_count,
        'hit_count': hit_count,
        'miss_count': miss_count,
        'fp_count': fp_count,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'hit_files': list(hit_files),
        'missed_files': list(missed_files),
        'fp_files': list(fp_files),
        'exclude_files': list(exclude_files)
    }
