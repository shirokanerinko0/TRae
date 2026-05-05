#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试不同配置组合的脚本
测试 config.json 中 code_snippet 配置的不同组合
"""

import os
import json
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, PROJECT_ROOT)

CONFIG_FILE = os.path.join(PROJECT_ROOT, 'config.json')

from src.JavaCodeAnalyzer.tree_sitter_java_analyzer import analyze_directory
from src.model.calculate_code_vectors import process_analysis_files
from src.trace_link.main import trace_links
from src.utils.utils import load_config, save_config


def save_original_config():
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def restore_original_config(original_config):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(original_config, f, indent=2, ensure_ascii=False)

def update_config(code_snippet):
    config = load_config()
    config['code_snippet'] = code_snippet
    save_config(config)

def run_trace_link():
    print("\n开始运行 trace_links()...")
    trace_links()
    print("trace_links() 执行完成")

def main():
    original_config = save_original_config()

    fixed_base_snippets = ["IO", "IMO"]

    try:
        combinations = [
            ["CO","MO"],
            ["CO","MCC"],
            ["CO","MDCC", "IMD"],
            ["CO","MCC","MC"],
            ["CO","MDCC","MC", "IMD"],
            ["CO","MCC","MO"],
            ["CO","MDCC","MO", "IMD"],
            ["CO","MCC","MC","MO"],
            ["CO","MDCC","MC","MO", "IMD"],
            ["CO","MDCC","MC","MCC","MD", "IMD"],
            ["CO","MCC","MD"],
            ["CO","MDCC", "IMD", "MD"],
            ["CO","MCC","MC","MD"],
            ["CO","MDCC","MC", "IMD","MD"],
            ["CO","MCC","MO","MD"],
            ["CO","MDCC","MO", "IMD","MD"],
            ["CO","MCC","MC","MO","MD"],
            ["CO","MDCC","MC","MO", "IMD","MD"],
            ["CO","MDCC","MC","MCC","MD", "IMD","MD"]
        ]

        all_tests = []
        for combo in combinations:
            combined = list(set(combo + fixed_base_snippets))
            all_tests.append(combined)

        print(f"总共需要测试 {len(all_tests)} 个配置组合\n")

        test_directory = f"data\\{load_config()['repo']}\\origin_src"

        print("=" * 80)
        print("预执行：代码分析和向量计算（只执行一次）")
        print("=" * 80)

        print("\n1. 分析代码结构...")
        analyze_directory(test_directory)

        print("\n2. 计算代码向量...")
        process_analysis_files(test_directory)

        print("\n" + "=" * 80)
        print("开始批量测试不同配置")
        print("=" * 80)

        for i, code_snippet in enumerate(all_tests, 1):
            print(f"\n{'=' * 80}")
            print(f"测试组合 {i}/{len(all_tests)}: {code_snippet}")
            print("=" * 80)

            update_config(code_snippet)
            run_trace_link()

        print("\n" + "=" * 80)
        print("所有测试完成！")
        print("=" * 80)

    finally:
        restore_original_config(original_config)
        print("\n已恢复原始配置")

    print("\n" + "=" * 80)
    print("运行分析程序...")
    print("=" * 80)
    from analyze_results import main as analyze_main
    analyze_main()

if __name__ == "__main__":
    main()
