#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据抽取与解析层，负责从GitHub API返回的数据中抽取和解析信息
"""

import os

# 尝试从main.py导入全局配置变量
try:
    from main import CONFIG
except ImportError:
    CONFIG = None


class DataExtractor:
    """
    数据抽取与解析类，提供从GitHub API返回的数据中抽取信息的方法
    """
    
    def __init__(self, limits=None):
        """
        初始化DataExtractor实例
        :param limits: 限制配置字典，包含max_commits等配置项
        """
        # 定义源代码文件扩展名
        if CONFIG and "source_code_extensions" in CONFIG:
            self.source_code_extensions = CONFIG["source_code_extensions"]
        else:
            self.source_code_extensions = {
                '.py', '.java', '.c', '.cpp'
            }
        print(f"源代码文件扩展名: {self.source_code_extensions}")
        # 设置默认限制
        self.limits = {
            'max_commits': 1000,
            'max_issues': 1000,
            'max_pull_requests': 1000,
            'max_files': 1000
        }
        
        # 首先使用配置文件中的limits值
        if CONFIG and "limits" in CONFIG:
            self.limits.update(CONFIG["limits"])
        
        # 如果提供了limits参数，更新默认值
        if limits:
            self.limits.update(limits)
    
    def extract_issues(self, issues):
        """
        从Issues迭代器中抽取信息
        :param issues: Issues迭代器
        :return: Issues列表
        """
        issues_list = []
        for issue in issues:
            try:
                issues_list.append({
                    "id": issue.id,
                    "number": issue.number,
                    "title": issue.title or "",
                    "body": issue.body or "",
                    "state": issue.state,
                    "created_at": issue.created_at,
                    "updated_at": issue.updated_at,
                    "labels": [label.name for label in issue.labels],
                    "user": issue.user.login if issue.user else "",
                    "assignee": issue.assignee.login if issue.assignee else None
                })
            except Exception as item_error:
                print(f"处理Issue #{issue.number} 时出错: {str(item_error)}")
                import traceback
                traceback.print_exc()
                continue
        return issues_list
    
    def extract_commits(self, commits):
        """
        从Commits迭代器中抽取信息
        :param commits: Commits迭代器
        :return: Commits列表
        """
        commits_list = []
        count = 0
        max_commits = self.limits.get('max_commits', 1000)  # 从限制配置中获取
        
        for commit in commits:
            try:
                # 检查是否超过限制
                if count >= max_commits:
                    print(f"已达到Commits获取限制 ({max_commits}个)")
                    break
                
                commit_data = {
                    "sha": commit.sha,
                    "message": commit.commit.message or "",
                    "url": commit.html_url
                }
                
                # 安全获取作者信息
                if commit.commit.author:
                    commit_data["author"] = commit.commit.author.name or ""
                    commit_data["email"] = commit.commit.author.email or ""
                    commit_data["date"] = commit.commit.author.date
                else:
                    commit_data["author"] = None
                    commit_data["email"] = None
                    commit_data["date"] = None
                
                # 安全获取提交者信息
                if commit.commit.committer:
                    commit_data["committer"] = commit.commit.committer.name or ""
                    commit_data["committer_email"] = commit.commit.committer.email or ""
                    commit_data["committer_date"] = commit.commit.committer.date
                else:
                    commit_data["committer"] = None
                    commit_data["committer_email"] = None
                    commit_data["committer_date"] = None
                
                commits_list.append(commit_data)
                count += 1
            except Exception as item_error:
                print(f"处理Commit {commit.sha} 时出错: {str(item_error)}")
                import traceback
                traceback.print_exc()
                continue
        
        return commits_list
    
    def extract_pull_requests(self, prs):
        """
        从Pull Requests迭代器中抽取信息
        :param prs: Pull Requests迭代器
        :return: Pull Requests列表
        """
        prs_list = []
        for pr in prs:
            try:
                prs_list.append({
                    "id": pr.id,
                    "number": pr.number,
                    "title": pr.title or "",
                    "body": pr.body or "",
                    "state": pr.state,
                    "created_at": pr.created_at,
                    "updated_at": pr.updated_at,
                    "merged_at": pr.merged_at,
                    "user": pr.user.login if pr.user else "",
                    "assignee": pr.assignee.login if pr.assignee else None,
                    "labels": [label.name for label in pr.labels],
                    "head": pr.head.ref,
                    "base": pr.base.ref,
                    "merged": pr.merged,
                    "merge_commit_sha": pr.merge_commit_sha
                })
            except Exception as item_error:
                print(f"处理PR #{pr.number} 时出错: {str(item_error)}")
                import traceback
                traceback.print_exc()
                continue
        return prs_list
    
    def extract_files(self, repo, path="."):
        """
        递归获取仓库中的所有文件
        :param repo: 仓库对象
        :param path: 起始路径
        :return: 文件列表
        """
        try:
            from src.api.github_api import GitHubAPI, DEBUG
            api = GitHubAPI("dummy_token", debug=DEBUG)  # 仅用于调用方法，不需要实际认证
            contents = api.get_contents(repo, path)
            files_list = []
            
            for content in contents:
                if content.type == "dir":
                    # 递归处理子目录
                    files_list.extend(self.extract_files(repo, content.path))
                else:
                    # 处理文件
                    files_list.append({
                        "path": content.path,
                        "name": content.name,
                        "type": content.type,
                        "size": content.size,
                        "download_url": content.download_url
                    })
            return files_list
        except Exception as e:
            print(f"获取文件失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def filter_source_files(self, files):
        """
        过滤文件列表，保留源代码文件
        :param files: 文件列表
        :return: 过滤后的源代码文件列表
        """
        source_files = []
        for file in files:
            # 获取文件扩展名
            _, ext = os.path.splitext(file['name'])
            # 检查是否为源代码文件
            if ext in self.source_code_extensions:
                source_files.append(file)
        return source_files
    
    def is_source_file(self, file_path):
        """
        检查指定文件是否为源代码文件
        :param file_path: 文件路径
        :return: 是否为源代码文件
        """
        _, ext = os.path.splitext(file_path)
        return ext in self.source_code_extensions
    
    def save_source_files(self, repo, target_dir):
        """
        按原仓库结构保存源代码文件
        :param repo: 仓库对象
        :param target_dir: 目标保存目录
        :return: 保存的文件数量
        """
        try:
            from src.api.github_api import GitHubAPI
            api = GitHubAPI("dummy_token")  # 仅用于调用方法，不需要实际认证
            
            # 确保目标目录存在
            os.makedirs(target_dir, exist_ok=True)
            
            # 递归保存文件
            saved_count = self._save_files_recursive(repo, api, ".", target_dir)
            
            return saved_count
        except Exception as e:
            print(f"保存源代码文件失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0
    
    def _save_files_recursive(self, repo, api, repo_path, target_dir):
        """
        递归保存文件的辅助方法
        :param repo: 仓库对象
        :param api: GitHubAPI实例
        :param repo_path: 仓库中的当前路径
        :param target_dir: 目标保存目录
        :return: 保存的文件数量
        """
        saved_count = 0
        
        try:
            contents = api.get_contents(repo, repo_path)
            
            for content in contents:
                if content.type == "dir":
                    # 递归处理子目录
                    sub_target_dir = os.path.join(target_dir, content.name)
                    os.makedirs(sub_target_dir, exist_ok=True)
                    saved_count += self._save_files_recursive(repo, api, content.path, sub_target_dir)
                else:
                    # 检查是否为源代码文件
                    if self.is_source_file(content.path):
                        try:
                            # 获取文件内容
                            file_content = api.get_file_content(repo, content.path)
                            if file_content is not None:
                                # 构建目标文件路径
                                target_file_path = os.path.join(target_dir, content.name)
                                
                                # 保存文件内容
                                with open(target_file_path, 'w', encoding='utf-8', errors='replace') as f:
                                    f.write(file_content)
                                
                                saved_count += 1
                                # 每保存10个文件打印一次进度
                                if saved_count % 10 == 0:
                                    print(f"  已保存 {saved_count} 个源代码文件...")
                        except Exception as e:
                            print(f"  保存文件 {content.path} 失败: {str(e)}")
                            continue
        except Exception as e:
            print(f"  处理目录 {repo_path} 失败: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return saved_count
