#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据预处理与清洗层，负责对抽取的数据进行预处理和清洗
"""

import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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
