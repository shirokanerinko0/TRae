#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
工具文件，包含配置文件加载和数据保存等功能
"""
import sys
import json
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

def load_config(config_file="config.json"):
    """
    从配置文件加载配置信息
    :param config_file: 配置文件路径
    :return: 配置字典
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"加载配置文件失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

CONFIG = load_config()

def save_data(data, file_path):
    """
    保存数据到指定文件
    :param data: 要保存的数据
    :param file_path: 文件路径
    """
    try:
        # 处理路径，确保使用正确的路径分隔符
        file_path = os.path.normpath(file_path)
        
        # 确保目录存在
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        # 保存数据
        with open(file_path, 'w', encoding='utf-8') as f:
            # 自定义序列化函数，处理复杂类型
            def default_serializer(obj):
                if hasattr(obj, '__dict__'):
                    return obj.__dict__
                elif isinstance(obj, (bytes, bytearray)):
                    return obj.decode('utf-8', errors='replace')
                else:
                    return str(obj)
            
            json.dump(data, f, ensure_ascii=False, indent=2, default=default_serializer)
        print(f"数据保存成功: {file_path}")
    except Exception as e:
        print(f"保存数据失败: {str(e)}")
        import traceback
        traceback.print_exc()


def save_config(config, config_file="config.json"):
    """
    保存配置到文件
    :param config: 配置字典
    :param config_file: 配置文件路径
    """
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"配置保存成功: {config_file}")
        return True
    except Exception as e:
        print(f"保存配置失败: {str(e)}")
        return False


def get_trace_link_result_file_name():
    global CONFIG
    CONFIG = load_config()
    encode_model_name= CONFIG['encode_model_name']
    top_k= CONFIG['top_k']
    # 处理top_k为数组的情况
    if isinstance(top_k, list):
        top_k_str = str(top_k[0])+'_'+str(top_k[-1])
    else:
        top_k_str = str(top_k)
    name = (
        'trace_link'
        +('_llm' if CONFIG['trace_link']['use_llm'] else '')
        +(CONFIG['SiliconFlow']['model'].replace('/', '_') if CONFIG['requirement_processing']['use_llm_processing'] else '')
        +(f'_{encode_model_name}' if encode_model_name else '')
        +(f'_top{top_k_str}' if top_k else '')
        )
    code_snippet_types = CONFIG.get("code_snippet", [])
    code_snippet_types.sort()
    if code_snippet_types:
        name = name + "_" + "_".join(code_snippet_types)
    return name + '.json'

def get_requirements_processed_file_name():
    name = (
    'requirements_processed'
    + ('_llm_' if CONFIG['requirement_processing']['use_llm_processing'] else '')
    + (CONFIG['SiliconFlow']['model'].replace('/', '_') if CONFIG['requirement_processing']['use_llm_processing'] else '')
    + '.json'
    )
    return name

def read_json_file(file_path):
    """
    读取json文件，返回List或dict
    :param file_path: json文件路径
    :return: json数据
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"读取json文件失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
