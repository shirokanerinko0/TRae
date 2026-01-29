#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GitHub API访问层，负责与GitHub API的交互
"""

from github import Github
import github.Auth
import json
import os
import logging
from datetime import datetime

# 尝试从main.py导入全局配置变量
try:
    from main import CONFIG
except ImportError:
    CONFIG = None

# 全局DEBUG变量，用于控制debug模式
DEBUG = True


class GitHubAPI:
    """
    GitHub API访问类，提供与GitHub API交互的方法
    """
    
    def __init__(self, access_token, debug=None, log_dir="logs"):
        """
        初始化GitHubAPI实例
        :param access_token: GitHub个人访问令牌
        :param debug: 是否开启debug模式，输出API调用详情。如果为None，则使用配置文件或全局DEBUG变量
        :param log_dir: 日志文件保存目录
        """
        self.access_token = access_token
        # 优先使用传入的debug参数
        if debug is not None:
            self.debug = debug
        # 其次使用配置文件中的debug值
        elif CONFIG and "debug" in CONFIG:
            self.debug = CONFIG["debug"]
        # 最后使用全局DEBUG变量
        else:
            self.debug = DEBUG
        self.log_dir = log_dir
        
        # 初始化日志
        if self.debug:
            self._init_logger()
        
        # 使用新的认证方式
        auth = github.Auth.Token(access_token)
        self.g = Github(auth=auth)
    
    def _init_logger(self):
        """
        初始化日志记录器
        """
        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 创建日志文件路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"github_api_debug_{timestamp}.log")
        
        # 配置日志
        logging.basicConfig(
            filename=log_file,
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            encoding='utf-8'
        )
        
        self.logger = logging.getLogger(__name__)
        print(f"Debug模式已开启，日志将保存到: {log_file}")
    
    def _log_api_response(self, api_url, response_data):
        """
        记录API响应到日志
        :param api_url: API请求URL
        :param response_data: API响应数据
        """
        if self.debug and hasattr(self, 'logger'):
            try:
                # 确保数据可序列化
                if isinstance(response_data, (list, dict)):
                    serialized_data = json.dumps(response_data, ensure_ascii=False, indent=2)
                else:
                    serialized_data = str(response_data)
                
                log_message = f"API URL: {api_url}\nResponse: {serialized_data}"
                self.logger.debug(log_message)
            except Exception as e:
                self.logger.error(f"记录API响应失败: {str(e)}")
    
    def get_repo(self, repo_owner, repo_name):
        """
        获取指定仓库
        :param repo_owner: 仓库所有者
        :param repo_name: 仓库名称
        :return: 仓库对象
        """
        api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
        
        try:
            repo = self.g.get_repo(f"{repo_owner}/{repo_name}")
            
            # 记录API响应
            if self.debug:
                repo_data = {
                    "name": repo.name,
                    "full_name": repo.full_name,
                    "description": repo.description,
                    "html_url": repo.html_url,
                    "stargazers_count": repo.stargazers_count,
                    "forks_count": repo.forks_count,
                    "open_issues_count": repo.open_issues_count
                }
                self._log_api_response(api_url, repo_data)
            
            return repo
        except Exception as e:
            print(f"获取仓库失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_issues(self, repo, state="all", labels=None):
        """
        获取仓库的Issues
        :param repo: 仓库对象
        :param state: Issues状态 (open, closed, all)
        :param labels: 标签列表，用于过滤
        :return: Issues迭代器
        """
        api_url = f"https://api.github.com/repos/{repo.owner.login}/{repo.name}/issues"
        params = f"?state={state}"
        if labels:
            params += f"&labels={','.join(labels)}"
        api_url += params
        
        try:
            if labels:
                issues = repo.get_issues(state=state, labels=labels)
            else:
                issues = repo.get_issues(state=state)
            
            # 记录API响应
            if self.debug:
                issues_list = list(issues)
                issues_data = []
                for issue in issues_list[:10]:  # 只记录前10个，避免日志过大
                    issues_data.append({
                        "number": issue.number,
                        "title": issue.title,
                        "state": issue.state,
                        "created_at": issue.created_at.isoformat() if issue.created_at else None,
                        "user": issue.user.login if issue.user else None,
                        "labels": [label.name for label in issue.labels]
                    })
                    print(issue)
                if len(issues_list) > 10:
                    issues_data.append({"message": f"... 等 {len(issues_list) - 10} 个issues"})
                self._log_api_response(api_url, issues_data)
                # 重新获取issues迭代器，因为之前的已经被消费
                if labels:
                    issues = repo.get_issues(state=state, labels=labels)
                else:
                    issues = repo.get_issues(state=state)
            
            return issues
        except Exception as e:
            print(f"获取Issues失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_commits(self, repo, since=None, until=None):
        """
        获取仓库的Commits
        :param repo: 仓库对象
        :param since: 开始日期
        :param until: 结束日期
        :return: Commits迭代器
        """
        api_url = f"https://api.github.com/repos/{repo.owner.login}/{repo.name}/commits"
        params = []
        if since:
            params.append(f"since={since.isoformat()}")
        if until:
            params.append(f"until={until.isoformat()}")
        if params:
            api_url += "?" + "&".join(params)
        
        try:
            kwargs = {}
            if since is not None:
                kwargs["since"] = since
            if until is not None:
                kwargs["until"] = until

            commits = repo.get_commits(**kwargs)
            
            # 记录API响应
            if self.debug:
                commits_list = list(commits)
                commits_data = []
                for commit in commits_list[:10]:  # 只记录前10个，避免日志过大
                    commits_data.append({
                        "sha": commit.sha,
                        "message": commit.commit.message,
                        "author": commit.commit.author.name if commit.commit.author else None,
                        "date": commit.commit.author.date.isoformat() if commit.commit.author and commit.commit.author.date else None
                    })
                if len(commits_list) > 10:
                    commits_data.append({"message": f"... 等 {len(commits_list) - 10} 个commits"})
                self._log_api_response(api_url, commits_data)
                # 重新获取commits迭代器，因为之前的已经被消费
                if since is not None and until is not None:
                    commits = repo.get_commits(since=since, until=until)
                elif since is not None:
                    commits = repo.get_commits(since=since)
                elif until is not None:
                    commits = repo.get_commits(until=until)
                else:
                    commits = repo.get_commits()
            
            return commits
        except Exception as e:
            print(f"获取Commits失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_pull_requests(self, repo, state="all"):
        """
        获取仓库的Pull Requests
        :param repo: 仓库对象
        :param state: PR状态 (open, closed, all)
        :return: Pull Requests迭代器
        """
        api_url = f"https://api.github.com/repos/{repo.owner.login}/{repo.name}/pulls?state={state}"
        
        try:
            pulls = repo.get_pulls(state=state)
            
            # 记录API响应
            if self.debug:
                pulls_list = list(pulls)
                pulls_data = []
                for pr in pulls_list[:10]:  # 只记录前10个，避免日志过大
                    pulls_data.append({
                        "number": pr.number,
                        "title": pr.title,
                        "state": pr.state,
                        "created_at": pr.created_at.isoformat() if pr.created_at else None,
                        "merged": pr.merged,
                        "merged_at": pr.merged_at.isoformat() if pr.merged_at else None,
                        "user": pr.user.login if pr.user else None,
                        "head": pr.head.ref,
                        "base": pr.base.ref
                    })
                if len(pulls_list) > 10:
                    pulls_data.append({"message": f"... 等 {len(pulls_list) - 10} 个pull requests"})
                self._log_api_response(api_url, pulls_data)
                # 重新获取pulls迭代器，因为之前的已经被消费
                pulls = repo.get_pulls(state=state)
            
            return pulls
        except Exception as e:
            print(f"获取Pull Requests失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_contents(self, repo, path="."):
        """
        获取仓库指定路径的内容
        :param repo: 仓库对象
        :param path: 路径
        :return: 内容对象
        """
        api_url = f"https://api.github.com/repos/{repo.owner.login}/{repo.name}/contents/{path}"
        
        try:
            contents = repo.get_contents(path)
            
            # 记录API响应
            if self.debug:
                contents_data = []
                for content in contents:
                    content_info = {
                        "name": content.name,
                        "path": content.path,
                        "type": content.type,
                        "size": content.size,
                        "download_url": content.download_url
                    }
                    contents_data.append(content_info)
                self._log_api_response(api_url, contents_data)
            
            return contents
        except Exception as e:
            print(f"获取内容失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_file_content(self, repo, file_path):
        """
        获取指定文件的内容
        :param repo: 仓库对象
        :param file_path: 文件路径
        :return: 文件内容
        """
        api_url = f"https://api.github.com/repos/{repo.owner.login}/{repo.name}/contents/{file_path}"
        
        try:
            content = repo.get_contents(file_path)
            
            # 获取文件内容
            if content.encoding == "base64":
                import base64
                file_content = base64.b64decode(content.content).decode('utf-8', errors='replace')
            else:
                file_content = content.content
            
            # 记录API响应（只记录文件信息，不记录文件内容，避免日志过大）
            if self.debug:
                file_info = {
                    "name": content.name,
                    "path": content.path,
                    "size": content.size,
                    "encoding": content.encoding,
                    "content_length": len(file_content) if file_content else 0
                }
                self._log_api_response(api_url, file_info)
            
            return file_content
        except Exception as e:
            print(f"获取文件内容失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
