#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试不同配置组合的脚本
测试 config.json 中 code_snippet 配置的不同组合
支持多仓库、多提示词测试
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

def update_repo_config(repo_name):
    config = load_config()
    config['repo'] = repo_name
    save_config(config)

def update_prompt_config(prompt_name):
    config = load_config()
    config['requirement_processing']['prompt_name'] = prompt_name
    if prompt_name == "noprompt":
        config['requirement_processing']['use_llm_processing'] = False
    else:
        config['requirement_processing']['use_llm_processing'] = True
    save_config(config)

def run_trace_link():
    from src.trace_link.main import trace_links
    from src.utils.utils import load_config
    import src.trace_link.main as trace_link_module
    import src.utils.utils as utils_module

    new_config = load_config()
    trace_link_module.CONFIG = new_config
    utils_module.CONFIG = new_config
    trace_link_module.encoder = None
    trace_link_module.data = None
    print("\n开始运行 trace_links()...")
    trace_links()
    print("trace_links() 执行完成")

def run_analyze_results():
    from src.utils.utils import load_config
    import analyze_results as analyze_module
    analyze_module.CONFIG = load_config()
    print("\n运行 analyze_results()...")
    from analyze_results import main as analyze_main
    analyze_main()

def test_repo_prompt_combinations(repo_name, combinations, fixed_base_snippets, prompts_to_test):
    print(f"\n{'#' * 80}")
    print(f"开始测试仓库: {repo_name}")
    print(f"{'#' * 80}")

    update_repo_config(repo_name)

    test_directory = f"data\\{repo_name}\\origin_src"

    print("\n1. 分析代码结构...")
    analyze_directory(test_directory)

    print("\n2. 计算代码向量...")
    process_analysis_files(test_directory)

    all_tests = []
    for combo in combinations:
        combined = sorted(list(set(combo + fixed_base_snippets)))
        all_tests.append(combined)

    print(f"代码片段组合数量: {len(all_tests)}")
    print(f"提示词数量: {len(prompts_to_test)}")
    print(f"总测试数: {len(all_tests) * len(prompts_to_test)}")

    for prompt_name in prompts_to_test:
        print(f"\n{'#' * 80}")
        print(f"提示词: {prompt_name}")
        print(f"{'#' * 80}")

        update_prompt_config(prompt_name)

        for i, code_snippet in enumerate(all_tests, 1):
            print(f"\n{'=' * 80}")
            print(f"提示词: {prompt_name} | 组合 {i}/{len(all_tests)}: {code_snippet}")
            print("=" * 80)

            update_config(code_snippet)
            run_trace_link()

        print("\n" + "=" * 80)
        print(f"提示词 {prompt_name} 所有组合测试完成！")
        print("=" * 80)

    print("\n" + "=" * 80)
    print(f"仓库 {repo_name} 所有测试完成！")
    print("=" * 80)

    run_analyze_results()

def main():
    original_config = save_original_config()

    fixed_base_snippets = ["CO", "IO", "IMO"]

    combinations = [
        ["MO"],
        ["MCC"],
        ["MD", "IMD"],
        ["MDCC", "IMD"],
        ["MO", "MCC"],
        ["MO", "MDCC", "IMD"],
        # ["MO", "MDCC", "MCC","IMD"],
    ]

    prompts_to_test = [
        "prompt2",
        # "prompt3",
        # "prompt4",
        # "prompt5",
        # "prompt6",
        # "prompt7",
        # "prompt8",
        # "prompt9",
        # "prompt10",
        # "prompt11",
        # "prompt12",
        # "prompt13",
        # "prompt14",
        # "prompt15",
        # "prompt16",
        # "prompt17",
        # "prompt18",
        # "prompt19",
        # "noprompt",
    ]

    repos_to_test = [
        "netty",
        "kafka",
        # "redisson",
    ]

    try:
        for repo in repos_to_test:
            test_repo_prompt_combinations(repo, combinations, fixed_base_snippets, prompts_to_test)

        print("\n" + "#" * 80)
        print("所有仓库测试完成！")
        print("#" * 80)

    finally:
        restore_original_config(original_config)
        print("\n已恢复原始配置")

if __name__ == "__main__":
    main()