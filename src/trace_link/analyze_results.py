import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.utils import load_config, get_trace_link_result_file_name, get_requirements_processed_file_name

import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
CONFIG = load_config()

class TraceLinkResultAnalyzer:
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.results_files = []
        self.all_results = []

    def load_all_results(self) -> List[Dict]:
        self.results_files = [
            f for f in os.listdir(self.results_dir)
            if f.startswith("trace_link") and f.endswith(".json")
        ]
        self.all_results = []
        for filename in self.results_files:
            filepath = os.path.join(self.results_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                config = self.parse_filename(filename)
                self.all_results.append({
                    'filename': filename,
                    'config': config,
                    'data': data
                })
        return self.all_results

    def parse_filename(self, filename: str) -> Dict:
        pro_match = re.match(r"trace_linkPro_(.+?)_(.+?)_top(\d+)_(\d+)(?:_(.+?))?\.json", filename)
        llm_no_pro_match = re.match(r"trace_link([a-zA-Z0-9_-]+?)_(jina_code|unixcoder|sbert|fastText)_top(\d+)_(\d+)(?:_(.+?))?\.json", filename)
        normal_match = re.match(r"trace_link([a-zA-Z0-9_-]+?)_top(\d+)_(\d+)(?:_(.+?))?\.json", filename)

        if pro_match:
            llm_name = "Pro_" + pro_match.group(1).replace("_", "/")
            encoder = pro_match.group(2)
            top_k_min = int(pro_match.group(3))
            top_k_max = int(pro_match.group(4))
            snippet_types = pro_match.group(5).split("_") if pro_match.group(5) else []
            return {
                'llm_model': llm_name,
                'encoder': encoder,
                'top_k_range': f"{top_k_min}-{top_k_max}",
                'top_k_min': top_k_min,
                'top_k_max': top_k_max,
                'snippet_types': snippet_types,
                'snippet_types_str': "_".join(snippet_types) if snippet_types else "default",
                'has_llm': True,
                'use_llm': llm_name
            }
        elif llm_no_pro_match:
            llm_name = llm_no_pro_match.group(1)
            encoder = llm_no_pro_match.group(2)
            top_k_min = int(llm_no_pro_match.group(3))
            top_k_max = int(llm_no_pro_match.group(4))
            snippet_types = llm_no_pro_match.group(5).split("_") if llm_no_pro_match.group(5) else []
            return {
                'llm_model': llm_name,
                'encoder': encoder,
                'top_k_range': f"{top_k_min}-{top_k_max}",
                'top_k_min': top_k_min,
                'top_k_max': top_k_max,
                'snippet_types': snippet_types,
                'snippet_types_str': "_".join(snippet_types) if snippet_types else "default",
                'has_llm': True,
                'use_llm': llm_name
            }
        elif normal_match:
            encoder = normal_match.group(1)
            top_k_min = int(normal_match.group(2))
            top_k_max = int(normal_match.group(3))
            snippet_types = normal_match.group(4).split("_") if normal_match.group(4) else []
            return {
                'llm_model': None,
                'encoder': encoder,
                'top_k_range': f"{top_k_min}-{top_k_max}",
                'top_k_min': top_k_min,
                'top_k_max': top_k_max,
                'snippet_types': snippet_types,
                'snippet_types_str': "_".join(snippet_types) if snippet_types else "default",
                'has_llm': False,
                'use_llm': "未使用LLM"
            }
        else:
            return {
                'llm_model': None,
                'encoder': 'unknown',
                'top_k_range': 'unknown',
                'top_k_min': 0,
                'top_k_max': 0,
                'snippet_types': [],
                'snippet_types_str': 'unknown',
                'has_llm': False,
                'use_llm': "未使用LLM"
            }

    def get_all_statistics(self) -> List[Dict]:
        rows = []
        for result in self.all_results:
            config = result['config']
            statistics = result['data'].get('statistics', {})

            for top_k, stats in statistics.items():
                hit_rate = stats.get('requirements_with_at_least_one_hit', 0) / stats.get('requirements_with_change_files', 1) if stats.get('requirements_with_change_files', 0) > 0 else 0.0
                rows.append({
                    'llm_model': config['llm_model'],
                    'encoder': config['encoder'],
                    'snippet_types': config['snippet_types_str'],
                    'top_k': int(top_k),
                    'has_llm': config.get('has_llm', False),
                    'use_llm': config.get('use_llm', '未使用LLM'),
                    'total_requirements': stats.get('total_requirements', 0),
                    'requirements_with_change_files': stats.get('requirements_with_change_files', 0),
                    'requirements_with_at_least_one_hit': stats.get('requirements_with_at_least_one_hit', 0),
                    'total_change_files': stats.get('total_change_files', 0),
                    'total_predicted_files': stats.get('total_predicted_files', 0),
                    'total_hit_files': stats.get('total_hit_files', 0),
                    'total_fp_files': stats.get('total_fp_files', 0),
                    'overall_recall': stats.get('overall_recall', 0.0),
                    'overall_precision': stats.get('overall_precision', 0.0),
                    'overall_f1': stats.get('overall_f1', 0.0),
                    'average_recall': stats.get('average_recall', 0.0),
                    'average_precision': stats.get('average_precision', 0.0),
                    'average_f1': stats.get('average_f1', 0.0),
                    'hit_rate': hit_rate
                })

        return rows

    def compare_encoders(self, top_k: int = 10) -> List[Dict]:
        stats = self.get_all_statistics()
        filtered = [s for s in stats if s['top_k'] == top_k]

        encoder_stats = defaultdict(list)
        for s in filtered:
            key = (s['encoder'], s['snippet_types'])
            encoder_stats[key].append(s)

        results = []
        for (encoder, snippet), stat_list in encoder_stats.items():
            avg_recall = sum(s['overall_recall'] for s in stat_list) / len(stat_list)
            avg_precision = sum(s['overall_precision'] for s in stat_list) / len(stat_list)
            avg_f1 = sum(s['overall_f1'] for s in stat_list) / len(stat_list)
            avg_avg_recall = sum(s['average_recall'] for s in stat_list) / len(stat_list)
            avg_avg_precision = sum(s['average_precision'] for s in stat_list) / len(stat_list)
            avg_avg_f1 = sum(s['average_f1'] for s in stat_list) / len(stat_list)
            avg_hit_rate = sum(s['hit_rate'] for s in stat_list) / len(stat_list)
            results.append({
                'encoder': encoder,
                'snippet_types': snippet,
                'overall_recall': round(avg_recall, 4),
                'overall_precision': round(avg_precision, 4),
                'overall_f1': round(avg_f1, 4),
                'average_recall': round(avg_avg_recall, 4),
                'average_precision': round(avg_avg_precision, 4),
                'average_f1': round(avg_avg_f1, 4),
                'hit_rate': round(avg_hit_rate, 4),
                'count': len(stat_list)
            })

        return sorted(results, key=lambda x: x['overall_f1'], reverse=True)

    def compare_snippet_types(self, top_k: int = 10) -> List[Dict]:
        stats = self.get_all_statistics()
        filtered = [s for s in stats if s['top_k'] == top_k]

        snippet_stats = defaultdict(list)
        for s in filtered:
            snippet_stats[s['snippet_types']].append(s)

        results = []
        for snippet, stat_list in snippet_stats.items():
            avg_recall = sum(s['overall_recall'] for s in stat_list) / len(stat_list)
            avg_precision = sum(s['overall_precision'] for s in stat_list) / len(stat_list)
            avg_f1 = sum(s['overall_f1'] for s in stat_list) / len(stat_list)
            avg_avg_recall = sum(s['average_recall'] for s in stat_list) / len(stat_list)
            avg_avg_precision = sum(s['average_precision'] for s in stat_list) / len(stat_list)
            avg_avg_f1 = sum(s['average_f1'] for s in stat_list) / len(stat_list)
            avg_hit_rate = sum(s['hit_rate'] for s in stat_list) / len(stat_list)
            results.append({
                'snippet_types': snippet,
                'overall_recall': round(avg_recall, 4),
                'overall_precision': round(avg_precision, 4),
                'overall_f1': round(avg_f1, 4),
                'average_recall': round(avg_avg_recall, 4),
                'average_precision': round(avg_avg_precision, 4),
                'average_f1': round(avg_avg_f1, 4),
                'hit_rate': round(avg_hit_rate, 4),
                'count': len(stat_list)
            })

        return sorted(results, key=lambda x: x['overall_f1'], reverse=True)

    def compare_top_k_performance(self) -> List[Dict]:
        stats = self.get_all_statistics()

        topk_stats = defaultdict(list)
        for s in stats:
            topk_stats[s['top_k']].append(s)

        results = []
        for top_k, stat_list in sorted(topk_stats.items()):
            avg_recall = sum(s['overall_recall'] for s in stat_list) / len(stat_list)
            avg_precision = sum(s['overall_precision'] for s in stat_list) / len(stat_list)
            avg_f1 = sum(s['overall_f1'] for s in stat_list) / len(stat_list)
            avg_avg_recall = sum(s['average_recall'] for s in stat_list) / len(stat_list)
            avg_avg_precision = sum(s['average_precision'] for s in stat_list) / len(stat_list)
            avg_avg_f1 = sum(s['average_f1'] for s in stat_list) / len(stat_list)
            avg_hit_rate = sum(s['hit_rate'] for s in stat_list) / len(stat_list)
            results.append({
                'top_k': top_k,
                'overall_recall': round(avg_recall, 4),
                'overall_precision': round(avg_precision, 4),
                'overall_f1': round(avg_f1, 4),
                'average_recall': round(avg_avg_recall, 4),
                'average_precision': round(avg_avg_precision, 4),
                'average_f1': round(avg_avg_f1, 4),
                'hit_rate': round(avg_hit_rate, 4)
            })

        return results

    def get_recall_by_requirement(self, req_id: str) -> List[Dict]:
        results = []
        for result in self.all_results:
            config = result['config']
            data = result['data']

            for req in data.get('results', []):
                if req['req_id'] == req_id:
                    recall_info = req.get('recall', {})
                    for top_k, stats in recall_info.items():
                        results.append({
                            'config': f"{config['encoder']}_{config['snippet_types_str']}",
                            'top_k': int(top_k),
                            'recall': stats.get('recall', 0.0),
                            'hit_count': stats.get('hit_count', 0),
                            'total_change_files': stats.get('total_change_files', 0)
                        })
        return results

    def print_summary(self):
        print("=" * 100)
        print("需求追踪链接结果分析")
        print("=" * 100)
        print(f"\n共加载 {len(self.results_files)} 个结果文件\n")

        print("文件列表:")
        for i, f in enumerate(self.results_files, 1):
            config = self.parse_filename(f)
            print(f"  {i}. {config['encoder']} | {config['snippet_types_str']} | {config['use_llm']}")

        all_stats = self.get_all_statistics()
        llm_files = [f for f in self.results_files if self.parse_filename(f).get('has_llm', False)]
        non_llm_files = [f for f in self.results_files if not self.parse_filename(f).get('has_llm', False)]

        print("\n" + "=" * 100)
        print("按是否使用LLM分类统计")
        print("=" * 100)
        print(f"  使用LLM: {len(llm_files)} 个结果文件")
        print(f"  未使用LLM: {len(non_llm_files)} 个结果文件")

        print("\n" + "=" * 100)
        print("使用LLM vs 未使用LLM 对比 (Top-10, 按F1排序)")
        print("=" * 100)
        for use_llm_label in ["使用LLM", "未使用LLM"]:
            stats_filtered = [s for s in all_stats if s['top_k'] == 10 and s['use_llm'] == use_llm_label]
            if not stats_filtered:
                continue
            llm_stats = defaultdict(list)
            for s in stats_filtered:
                key = (s['encoder'], s['snippet_types'])
                llm_stats[key].append(s)
            results = []
            for (encoder, snippet), stat_list in llm_stats.items():
                avg_f1 = sum(s['overall_f1'] for s in stat_list) / len(stat_list)
                avg_recall = sum(s['overall_recall'] for s in stat_list) / len(stat_list)
                avg_precision = sum(s['overall_precision'] for s in stat_list) / len(stat_list)
                results.append({
                    'encoder': encoder,
                    'snippet_types': snippet,
                    'overall_recall': round(avg_recall, 4),
                    'overall_precision': round(avg_precision, 4),
                    'overall_f1': round(avg_f1, 4)
                })
            results.sort(key=lambda x: x['overall_f1'], reverse=True)
            print(f"\n--- {use_llm_label} ---")
            header = f"{'Encoder':>15} | {'Snippet Types':>25} | {'Recall':>8} | {'Precision':>10} | {'F1':>8}"
            print(header)
            print("-" * 80)
            for row in results[:10]:
                print(f"{row['encoder']:>15} | {row['snippet_types']:>25} | {row['overall_recall']:>8.4f} | {row['overall_precision']:>10.4f} | {row['overall_f1']:>8.4f}")

        print("\n" + "=" * 100)
        print("Top-K 性能对比 (各配置平均)")
        print("=" * 100)
        df_topk = self.compare_top_k_performance()
        header = f"{'Top-K':>6} | {'Recall':>8} | {'Precision':>10} | {'F1':>8} | {'Avg Recall':>10} | {'Avg Prec':>10} | {'Avg F1':>8} | {'Hit Rate':>9}"
        print(header)
        print("-" * 100)
        for row in df_topk:
            print(f"{row['top_k']:>6} | {row['overall_recall']:>8.4f} | {row['overall_precision']:>10.4f} | {row['overall_f1']:>8.4f} | {row['average_recall']:>10.4f} | {row['average_precision']:>10.4f} | {row['average_f1']:>8.4f} | {row['hit_rate']:>9.4f}")

        print("\n" + "=" * 100)
        print("编码器性能对比 (Top-10, 按F1排序)")
        print("=" * 100)
        df_encoder = self.compare_encoders(top_k=10)
        header = f"{'Encoder':>25} | {'Snippet Types':>25} | {'Recall':>8} | {'Precision':>10} | {'F1':>8} | {'Avg F1':>8} | {'Hit Rate':>9}"
        print(header)
        print("-" * 100)
        for row in df_encoder:
            print(f"{row['encoder']:>25} | {row['snippet_types']:>25} | {row['overall_recall']:>8.4f} | {row['overall_precision']:>10.4f} | {row['overall_f1']:>8.4f} | {row['average_f1']:>8.4f} | {row['hit_rate']:>9.4f}")

        print("\n" + "=" * 100)
        print("代码片段类型性能对比 (Top-10, 按F1排序)")
        print("=" * 100)
        df_snippet = self.compare_snippet_types(top_k=10)
        header = f"{'Snippet Types':>25} | {'Recall':>8} | {'Precision':>10} | {'F1':>8} | {'Avg F1':>8} | {'Hit Rate':>9}"
        print(header)
        print("-" * 100)
        for row in df_snippet:
            print(f"{row['snippet_types']:>25} | {row['overall_recall']:>8.4f} | {row['overall_precision']:>10.4f} | {row['overall_f1']:>8.4f} | {row['average_f1']:>8.4f} | {row['hit_rate']:>9.4f}")

        print("\n" + "=" * 100)
        print("各 Top-K 配置详细结果 (按F1排序)")
        print("=" * 100)
        for top_k in sorted(set(s['top_k'] for s in all_stats)):
            df_k = [s for s in all_stats if s['top_k'] == top_k]
            df_k.sort(key=lambda x: x['overall_f1'], reverse=True)
            print(f"\n--- Top-{top_k} ---")
            header = f"{'Snippet Types':>25} | {'UseLLM':>8} | {'Recall':>8} | {'Precision':>10} | {'F1':>8} | {'Avg F1':>8} | {'Hit Rate':>9}"
            print(header)
            print("-" * 100)
            for row in df_k:
                print(f"{row['snippet_types']:>25} | {row['use_llm']:>8} | {row['overall_recall']:>8.4f} | {row['overall_precision']:>10.4f} | {row['overall_f1']:>8.4f} | {row['average_f1']:>8.4f} | {row['hit_rate']:>9.4f}")

    def export_to_json(self, output_path: str):
        export_data = {
            'summary': {
                'total_files': len(self.results_files),
                'files': [self.parse_filename(f) for f in self.results_files]
            },
            'top_k_comparison': self.compare_top_k_performance(),
            'encoder_comparison_top10': self.compare_encoders(top_k=10),
            'snippet_comparison_top10': self.compare_snippet_types(top_k=10),
            'all_statistics': self.get_all_statistics()
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"\n结果已导出到: {output_path}")

    def export_to_excel(self, output_path: str):
        try:
            import openpyxl
            from openpyxl.styles import Font, Alignment, PatternFill
            from openpyxl.utils import get_column_letter
        except ImportError:
            print("请先安装 openpyxl: pip install openpyxl")
            return

        wb = openpyxl.Workbook()
        wb.remove(wb.active)

        header_font = Font(bold=True)
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_font_white = Font(bold=True, color="FFFFFF")

        all_stats = self.get_all_statistics()

        ws_summary = wb.create_sheet("Top-K汇总")
        headers_summary = ["Top-K", "Recall", "Precision", "F1", "Avg Recall", "Avg Precision", "Avg F1", "Hit Rate"]
        ws_summary.append(headers_summary)
        for col in range(1, len(headers_summary) + 1):
            cell = ws_summary.cell(row=1, column=col)
            cell.font = header_font_white
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal='center')
        for row in self.compare_top_k_performance():
            ws_summary.append([
                row['top_k'],
                f"{row['overall_recall']:.2%}",
                f"{row['overall_precision']:.2%}",
                f"{row['overall_f1']:.2%}",
                f"{row['average_recall']:.2%}",
                f"{row['average_precision']:.2%}",
                f"{row['average_f1']:.2%}",
                f"{row['hit_rate']:.2%}"
            ])
        for col in range(1, len(headers_summary) + 1):
            ws_summary.column_dimensions[get_column_letter(col)].width = 12

        ws_encoder = wb.create_sheet("编码器对比")
        headers_encoder = ["Encoder", "Snippet Types", "Recall", "Precision", "F1", "Avg F1", "Hit Rate"]
        ws_encoder.append(headers_encoder)
        for col in range(1, len(headers_encoder) + 1):
            cell = ws_encoder.cell(row=1, column=col)
            cell.font = header_font_white
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal='center')
        for row in self.compare_encoders(top_k=10):
            ws_encoder.append([
                row['encoder'],
                row['snippet_types'],
                f"{row['overall_recall']:.2%}",
                f"{row['overall_precision']:.2%}",
                f"{row['overall_f1']:.2%}",
                f"{row['average_f1']:.2%}",
                f"{row['hit_rate']:.2%}"
            ])
        ws_encoder.column_dimensions['A'].width = 15
        ws_encoder.column_dimensions['B'].width = 30

        ws_snippet = wb.create_sheet("片段类型对比")
        headers_snippet = ["Snippet Types", "Recall", "Precision", "F1", "Avg F1", "Hit Rate"]
        ws_snippet.append(headers_snippet)
        for col in range(1, len(headers_snippet) + 1):
            cell = ws_snippet.cell(row=1, column=col)
            cell.font = header_font_white
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal='center')
        for row in self.compare_snippet_types(top_k=10):
            ws_snippet.append([
                row['snippet_types'],
                f"{row['overall_recall']:.2%}",
                f"{row['overall_precision']:.2%}",
                f"{row['overall_f1']:.2%}",
                f"{row['average_f1']:.2%}",
                f"{row['hit_rate']:.2%}"
            ])
        ws_snippet.column_dimensions['A'].width = 30

        ws_llm_summary = wb.create_sheet("LLM对比")
        headers_llm = ["LLM类型", "Encoder", "Snippet Types", "Recall", "Precision", "F1"]
        ws_llm_summary.append(headers_llm)
        for col in range(1, len(headers_llm) + 1):
            cell = ws_llm_summary.cell(row=1, column=col)
            cell.font = header_font_white
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal='center')
        for use_llm_label in ["使用LLM", "未使用LLM"]:
            stats_filtered = [s for s in all_stats if s['top_k'] == 10 and s['use_llm'] == use_llm_label]
            if not stats_filtered:
                continue
            llm_stats = defaultdict(list)
            for s in stats_filtered:
                key = (s['encoder'], s['snippet_types'])
                llm_stats[key].append(s)
            for (encoder, snippet), stat_list in llm_stats.items():
                avg_f1 = sum(s['overall_f1'] for s in stat_list) / len(stat_list)
                avg_recall = sum(s['overall_recall'] for s in stat_list) / len(stat_list)
                avg_precision = sum(s['overall_precision'] for s in stat_list) / len(stat_list)
                ws_llm_summary.append([
                    use_llm_label,
                    encoder,
                    snippet,
                    f"{avg_recall:.2%}",
                    f"{avg_precision:.2%}",
                    f"{avg_f1:.2%}"
                ])
        ws_llm_summary.column_dimensions['A'].width = 12
        ws_llm_summary.column_dimensions['B'].width = 15
        ws_llm_summary.column_dimensions['C'].width = 30

        for top_k in sorted(set(s['top_k'] for s in all_stats)):
            df_k = [s for s in all_stats if s['top_k'] == top_k]
            df_k.sort(key=lambda x: x['overall_f1'], reverse=True)
            ws = wb.create_sheet(f"Top{top_k}详情")
            headers = ["Snippet Types", "UseLLM", "Recall", "Precision", "F1", "Avg F1", "Hit Rate",
                      "总需求", "有变更", "至少命中", "变更文件", "预测文件", "命中", "FP"]
            ws.append(headers)
            for col in range(1, len(headers) + 1):
                cell = ws.cell(row=1, column=col)
                cell.font = header_font_white
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal='center')
            for row in df_k:
                ws.append([
                    row['snippet_types'],
                    row['use_llm'],
                    f"{row['overall_recall']:.2%}",
                    f"{row['overall_precision']:.2%}",
                    f"{row['overall_f1']:.2%}",
                    f"{row['average_f1']:.2%}",
                    f"{row['hit_rate']:.2%}",
                    row['total_requirements'],
                    row['requirements_with_change_files'],
                    row['requirements_with_at_least_one_hit'],
                    row['total_change_files'],
                    row['total_predicted_files'],
                    row['total_hit_files'],
                    row['total_fp_files']
                ])
            ws.column_dimensions['A'].width = 30
            ws.column_dimensions['B'].width = 12

        wb.save(output_path)
        print(f"\nExcel结果已导出到: {output_path}")


def main():
    repo_name = CONFIG['repo']
    results_dir = f"data\\{repo_name}\\trace_link_results"
    json_output = os.path.join(results_dir, f"{repo_name}_analysis_output.json")
    excel_output = os.path.join(results_dir, f"{repo_name}_analysis_output.xlsx")

    analyzer = TraceLinkResultAnalyzer(results_dir)
    analyzer.load_all_results()
    analyzer.print_summary()
    analyzer.export_to_json(json_output)
    analyzer.export_to_excel(excel_output)


if __name__ == "__main__":
    main()