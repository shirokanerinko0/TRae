#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
主文件，整合各层功能模块
"""

from src.utils.utils import load_config, save_data
from src.api.github_api import GitHubAPI
from src.extractor.data_extractor import DataExtractor
from src.preprocessor.data_preprocessor import DataPreprocessor

# 全局配置变量
CONFIG = None


def main():
    """
    主函数，整合各层功能模块
    """
    global CONFIG
    # 从配置文件加载配置
    CONFIG = load_config()
    if not CONFIG:
        print("无法加载配置文件，请检查config.json文件是否存在且格式正确")
        return
    
    # 从配置中获取信息
    access_token = CONFIG.get("token")
    repo_owner = CONFIG.get("owner")
    repo_name = CONFIG.get("repo")
    
    # 验证配置信息
    if not all([access_token, repo_owner, repo_name]):
        print("配置信息不完整，请检查config.json文件中的token、owner和repo字段")
        return
    
    print(f"从配置文件加载信息成功: {repo_owner}/{repo_name}")
    
    # 初始化各层实例
    # GitHubAPI会从配置文件中读取debug值
    github_api = GitHubAPI(access_token)
    # DataExtractor会从配置文件中读取limits值
    extractor = DataExtractor()
    preprocessor = DataPreprocessor()
    
    # 获取仓库对象
    repo = github_api.get_repo(repo_owner, repo_name)
    if not repo:
        print("无法获取仓库，请检查输入信息和访问令牌权限")
        return
    
    print(f"成功获取仓库: {repo.name}")
    
    # 数据保存目录
    data_dir = f"data/{repo_name}"
    
    # 1. 采集Issues数据
    print("\n开始采集Issues数据...")
    issues = github_api.get_issues(repo, state="all")
    issues_list = extractor.extract_issues(issues)
    print(f"采集到 {len(issues_list)} 个Issues")
    
    # 预处理Issues数据
    processed_issues = preprocessor.preprocess_issues(issues_list)
    print(f"预处理完成 {len(processed_issues)} 个Issues")
    
    # 保存Issues数据
    save_data(issues_list, f"{data_dir}/issues_raw.json")
    save_data(processed_issues, f"{data_dir}/issues_processed.json")
    
    # 2. 采集Commits数据
    print("\n开始采集Commits数据...")
    commits = github_api.get_commits(repo)
    commits_list = extractor.extract_commits(commits)
    print(f"采集到 {len(commits_list)} 个Commits")
    
    # 预处理Commits数据
    processed_commits = preprocessor.preprocess_commits(commits_list)
    print(f"预处理完成 {len(processed_commits)} 个Commits")
    
    # 保存Commits数据
    save_data(commits_list, f"{data_dir}/commits_raw.json")
    save_data(processed_commits, f"{data_dir}/commits_processed.json")
    
    # 3. 采集Pull Requests数据
    print("\n开始采集Pull Requests数据...")
    prs = github_api.get_pull_requests(repo, state="all")
    prs_list = extractor.extract_pull_requests(prs)
    print(f"采集到 {len(prs_list)} 个Pull Requests")
    
    # 预处理Pull Requests数据
    processed_prs = preprocessor.preprocess_pull_requests(prs_list)
    print(f"预处理完成 {len(processed_prs)} 个Pull Requests")
    
    # 保存Pull Requests数据
    save_data(prs_list, f"{data_dir}/pull_requests_raw.json")
    save_data(processed_prs, f"{data_dir}/pull_requests_processed.json")
    
    # 4. 采集文件数据
    
    # 按原仓库结构保存源代码文件
    print("\n开始保存源代码文件到origin_src目录...")
    origin_src_dir = f"{data_dir}/origin_src"
    saved_count = extractor.save_source_files(repo, origin_src_dir)
    print(f"成功保存 {saved_count} 个源代码文件到 {origin_src_dir}")
    
    
    print("\n数据采集、预处理和保存完成！")


if __name__ == "__main__":
    main()
