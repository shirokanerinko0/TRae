#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据预处理与清洗层，负责对抽取的数据进行预处理和清洗
"""
import os
import sys
import time
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import nltk
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from src.utils.utils import load_config
import json
CONFIG = load_config()
use_llm_processing = CONFIG["requirement_processing"]["use_llm_processing"]

# 导入LLM处理函数
if use_llm_processing:
    from src.LLMapi.LLM_tset import process_requirement_text_llm

# 确保nltk停用词数据已下载
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# 跳过punkt资源的检查，因为punkt_tab可以替代它
# 避免punkt资源文件损坏导致的错误
try:
    # 尝试使用punkt_tab进行分词
    from nltk.tokenize import word_tokenize
    test_text = "This is a test."
    word_tokenize(test_text)
except Exception as e:
    print(f"分词初始化失败，但将继续执行: {str(e)}")


class DataPreprocessor:
    """
    数据预处理与清洗类，提供对抽取的数据进行预处理的方法
    """
    
    def __init__(self):
        """
        初始化DataPreprocessor实例
        """
        ## 初始化停用词列表
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        """
        预处理文本，包括去噪、停用词过滤和格式统一
        :param text: 原始文本
        :return: 预处理后的文本
        """
        if not text:
            return ""
        
        # 1. 去除特殊符号和链接
        # 去除链接
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'https\S+', '', text)

        
        # 2. 转换为小写
        text = text.lower()
        
        # 3. 分词
        try:
            tokens = word_tokenize(text)
        except Exception as e:
            print(f"分词失败: {str(e)}")
            return ""
        
        # 4. 过滤停用词
        filtered_tokens = [token for token in tokens if token not in self.stop_words]
        
        # 5. 格式统一，合并为空格分隔的字符串
        processed_text = ' '.join(filtered_tokens)
        
        return processed_text
    
    def get_tokens(self, text):
        """
        获取文本的分词结果
        :param text: 原始文本
        :return: 分词列表
        """
        if not text:
            return []
        
        # 预处理文本
        processed_text = self.preprocess_text(text)
        
        # 分词
        try:
            tokens = word_tokenize(processed_text)
        except Exception as e:
            print(f"分词失败: {str(e)}")
            return []
        
        # 过滤停用词
        filtered_tokens = [token for token in tokens if token not in self.stop_words]
        
        return filtered_tokens
    
    
    def preprocess_requirement(self, requirement):
        """
        预处理单个需求数据
        :param requirement: 需求数据
        :return: 预处理后的需求数据
        """
        processed_req = requirement.copy()
        
        # 1. 生成text_clean
        title = requirement.get('title', '')
        description = requirement.get('description', '')
        full_text = f"{title}\n{description}"
        processed_req['text_clean'] = self.preprocess_text(full_text)
        
        # 2. 生成tokens
        # processed_req['tokens'] = self.get_tokens(full_text)
        
        # 3. 使用LLM处理需求文本
        if use_llm_processing :
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    req_id = requirement.get('req_id', '')
                    print(f"使用LLM处理需求 {req_id}... (尝试 {retry_count + 1}/{max_retries})")
                    llm_result = process_requirement_text_llm(title, description)
                    llm_data = json.loads(llm_result)
                    
                    # 添加LLM处理结果字段
                    processed_req["llm_category"] = llm_data.get("category")
                    search_query = llm_data.get("search_query")
                    if search_query:
                        processed_req["search_query"] = search_query
                    else:
                        processed_req["search_query"] = full_text
                    processed_req["llm_reason"] = llm_data.get("reason")
                    processed_req["type"] = llm_data.get("category", "default")
                    break  # 处理成功，跳出循环
                except Exception as llm_error:
                    retry_count += 1
                    print(f"LLM处理需求 {req_id} 时出错: {str(llm_error)}")
                    if retry_count < max_retries:
                        print(f"2秒后重试...")
                        time.sleep(2)
                    else:
                        print(f"已达到最大重试次数，使用默认文本")
                        import traceback
                        traceback.print_exc()
                        processed_req["llm_reason"] = "error"
                        processed_req["llm_category"] = "error"
                        processed_req["search_query"] = full_text
                        processed_req["type"] = "default"
        else:
            processed_req["search_query"] =  description
            processed_req["type"] = "default"

        return processed_req
    
    def preprocess_requirements(self, requirements):
        """
        预处理需求列表（并行处理）
        :param requirements: 需求列表
        :return: 预处理后的需求列表
        """
        max_workers = CONFIG["requirement_processing"].get("max_workers", 8)
        # 过滤没有change_files的需求
        if CONFIG.get("filter_req_no_change_files", True):
            requirements=[req for req in requirements if req.get('change_files', [])]
        
        # 过滤change_files数量超过最大值的需求
        max_change_files = CONFIG.get("max_change_files", 20)
        requirements = [req for req in requirements if len(req.get('change_files', [])) <= max_change_files]
        
        processed_requirements = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_req = {
                executor.submit(self.preprocess_requirement, req): req 
                for req in requirements
            }
            
            for future in as_completed(future_to_req):
                processed_req = future.result()
                
                
                processed_requirements.append(processed_req)
        
        return processed_requirements
    
    def preprocess_issues(self, issues):
        """
        预处理Issues列表
        :param issues: Issues列表
        :return: 预处理后的Issues列表
        """
        processed_issues = []
        for issue in issues:
            processed_issue = issue.copy()
            processed_issue['title'] = self.preprocess_text(issue['title'])
            processed_issue['body'] = self.preprocess_text(issue['body'])
            processed_issues.append(processed_issue)
        return processed_issues
    
    def preprocess_commits(self, commits):
        """
        预处理Commits列表
        :param commits: Commits列表
        :return: 预处理后的Commits列表
        """
        processed_commits = []
        for commit in commits:
            processed_commit = commit.copy()
            processed_commit['message'] = self.preprocess_text(commit['message'])
            processed_commits.append(processed_commit)
        return processed_commits
    
    def preprocess_pull_requests(self, prs):
        """
        预处理Pull Requests列表
        :param prs: Pull Requests列表
        :return: 预处理后的Pull Requests列表
        """
        processed_prs = []
        for pr in prs:
            processed_pr = pr.copy()
            processed_pr['title'] = self.preprocess_text(pr['title'])
            processed_pr['body'] = self.preprocess_text(pr['body'])
            processed_prs.append(processed_pr)
        return processed_prs
