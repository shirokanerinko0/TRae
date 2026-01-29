#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GitHub API测试文件
"""

import unittest
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.api.github_api import GitHubAPI
from main import CONFIG
from src.utils.utils import load_config


class TestGitHubAPI(unittest.TestCase):
    """
    GitHub API测试类
    """
    
    @classmethod
    def setUpClass(cls):
        """
        测试前的初始化工作
        """
        # 加载配置文件
        if not CONFIG:
            cls.config = load_config()
        else:
            cls.config = CONFIG
        
        # 从配置中获取信息
        cls.access_token = cls.config.get("token")
        cls.repo_owner = cls.config.get("owner")
        cls.repo_name = cls.config.get("repo")
        
        # 初始化GitHubAPI实例
        cls.github_api = GitHubAPI(cls.access_token)
    
    def test_get_repo(self):
        """
        测试获取仓库功能
        """
        print("\n测试获取仓库功能...")
        repo = self.github_api.get_repo(self.repo_owner, self.repo_name)
        self.assertIsNotNone(repo, "无法获取仓库")
        self.assertEqual(repo.name, self.repo_name, "仓库名称不匹配")
        self.assertEqual(repo.owner.login, self.repo_owner, "仓库所有者不匹配")
        print(f"成功获取仓库: {repo.name}")
    
    def test_get_issues(self):
        """
        测试获取Issues功能
        """
        print("\n测试获取Issues功能...")
        repo = self.github_api.get_repo(self.repo_owner, self.repo_name)
        self.assertIsNotNone(repo, "无法获取仓库")
        
        issues = self.github_api.get_issues(repo, state="all")
        self.assertIsNotNone(issues, "无法获取Issues") 
        # 转换为列表并检查
        issues_list = list(issues)
        print(f"获取到 {len(issues_list)} 个Issues")
        self.assertGreaterEqual(len(issues_list), 0, "Issues列表为空")
    
    def test_get_commits(self):
        """
        测试获取Commits功能
        """
        print("\n测试获取Commits功能...")
        repo = self.github_api.get_repo(self.repo_owner, self.repo_name)
        self.assertIsNotNone(repo, "无法获取仓库")
        
        commits = self.github_api.get_commits(repo)
        self.assertIsNotNone(commits, "无法获取Commits")
        
        # 转换为列表并检查
        commits_list = list(commits)
        print(f"获取到 {len(commits_list)} 个Commits")
        self.assertGreaterEqual(len(commits_list), 0, "Commits列表为空")
    
    def test_get_pull_requests(self):
        """
        测试获取Pull Requests功能
        """
        print("\n测试获取Pull Requests功能...")
        repo = self.github_api.get_repo(self.repo_owner, self.repo_name)
        self.assertIsNotNone(repo, "无法获取仓库")
        
        prs = self.github_api.get_pull_requests(repo, state="all")
        self.assertIsNotNone(prs, "无法获取Pull Requests")
        
        # 转换为列表并检查
        prs_list = list(prs)
        print(f"获取到 {len(prs_list)} 个Pull Requests")
        self.assertGreaterEqual(len(prs_list), 0, "Pull Requests列表为空")
    
    def test_get_contents(self):
        """
        测试获取仓库内容功能
        """
        print("\n测试获取仓库内容功能...")
        repo = self.github_api.get_repo(self.repo_owner, self.repo_name)
        self.assertIsNotNone(repo, "无法获取仓库")
        
        contents = self.github_api.get_contents(repo, ".")
        self.assertIsNotNone(contents, "无法获取仓库内容")
        self.assertGreaterEqual(len(contents), 0, "仓库内容为空")
        
        print(f"获取到 {len(contents)} 个文件/目录")
        # 打印前5个内容
        for content in contents[:5]:
            print(f"  - {content.name} ({content.type})")
    
    def test_get_file_content(self):
        """
        测试获取文件内容功能
        """
        print("\n测试获取文件内容功能...")
        repo = self.github_api.get_repo(self.repo_owner, self.repo_name)
        self.assertIsNotNone(repo, "无法获取仓库")
        
        # 获取根目录内容，找到一个文件
        contents = self.github_api.get_contents(repo, ".")
        file_path = None
        
        for content in contents:
            if content.type == "file":
                file_path = content.path
                break
        
        self.assertIsNotNone(file_path, "未找到文件")
        
        # 获取文件内容
        file_content = self.github_api.get_file_content(repo, file_path)
        self.assertIsNotNone(file_content, f"无法获取文件内容: {file_path}")
        self.assertGreater(len(file_content), 0, f"文件内容为空: {file_path}")
        
        print(f"成功获取文件内容: {file_path}")
        print(f"文件长度: {len(file_content)} 字符")
        # 打印文件前100个字符
        print(f"文件前100个字符: {file_content[:100]}...")


if __name__ == '__main__':
    """
    运行测试
    """
    unittest.main()
