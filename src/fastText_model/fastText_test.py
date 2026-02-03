import fasttext
import fasttext.util
import numpy as np
import os,sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.utils import load_config
CONFIG = load_config()
if not os.path.exists(CONFIG["fastText"]["model_path"]):
    fasttext.util.download_model('en', if_exists='ignore')
    ##删除当前目录下的cc.en.300.bin.gz
    os.remove(CONFIG["fastText"]["model_name"]+".gz")
    ##下载的模型移动到model_data目录下
    os.replace(CONFIG["fastText"]["model_name"], CONFIG["fastText"]["model_path"])

model = fasttext.load_model(CONFIG["fastText"]["model_path"])
print(model.get_word_vector("apple")[:10])

def sentence_to_word_vectors(sentence, model, normalize=False):
    """
    将句子转换为词向量集合
    
    Args:
        sentence (str): 输入句子
        model (fasttext.FastText._FastText): fastText模型
        normalize (bool, optional): 是否对词向量进行归一化. Defaults to False.
    
    Returns:
        list: 词向量集合，每个元素是一个单词的词向量
    """
    # 简单分词（可以根据需要使用更复杂的分词方法）
    words = sentence.lower().split()
    
    # 获取每个单词的词向量
    word_vectors = []
    for word in words:
        try:
            vector = model.get_word_vector(word)
            if normalize:
                # 归一化向量
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
            word_vectors.append(vector)
        except Exception as e:
            print(f"获取单词 '{word}' 的词向量时出错: {e}")
            # 对于无法获取词向量的单词，可以选择跳过或使用零向量
            # 这里选择跳过
            pass
    
    return word_vectors

def sentence_to_average_vector(sentence, model, normalize=True):
    """
    将句子转换为平均词向量
    
    Args:
        sentence (str): 输入句子
        model (fasttext.FastText._FastText): fastText模型
        normalize (bool, optional): 是否对结果向量进行归一化. Defaults to True.
    
    Returns:
        numpy.ndarray: 句子的平均词向量
    """
    word_vectors = sentence_to_word_vectors(sentence, model, normalize=False)
    
    if not word_vectors:
        # 如果没有有效的词向量，返回零向量
        vector_dim = model.get_dimension()
        return np.zeros(vector_dim)
    
    # 计算平均向量
    average_vector = np.mean(word_vectors, axis=0)
    
    if normalize:
        norm = np.linalg.norm(average_vector)
        if norm > 0:
            average_vector = average_vector / norm
    
    return average_vector

# 测试句子转词向量功能
test_sentence = "I love eating apples and bananas"
print(f"测试句子: {test_sentence}")

# 获取词向量集合
word_vectors = sentence_to_word_vectors(test_sentence, model)
print(f"词向量数量: {len(word_vectors)}")
print(f"每个词向量维度: {len(word_vectors[0]) if word_vectors else 0}")

# 获取句子平均向量
average_vector = sentence_to_average_vector(test_sentence, model)
print(f"句子平均向量维度: {len(average_vector)}")
print(f"句子平均向量前10维: {average_vector[:10]}")

def calculate_sentence_similarity(sentence1, sentence2, model, method="cosine"):
    """
    计算两个句子之间的相似度
    
    Args:
        sentence1 (str): 第一个句子
        sentence2 (str): 第二个句子
        model (fasttext.FastText._FastText): fastText模型
        method (str, optional): 相似度计算方法. Defaults to "cosine".
    
    Returns:
        float: 两个句子之间的相似度得分
    """
    # 获取两个句子的平均词向量
    vec1 = sentence_to_average_vector(sentence1, model)
    vec2 = sentence_to_average_vector(sentence2, model)
    
    if method == "cosine":
        # 计算余弦相似度
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 > 0 and norm2 > 0:
            return dot_product / (norm1 * norm2)
        else:
            return 0.0
    else:
        raise ValueError(f"不支持的相似度计算方法: {method}")

# 测试不同句子的向量相似度
print("\n=== 测试句子相似度（平均词向量余弦相似度） ===")

# 测试相似句子对
sentence1 = "I love eating apples"
sentence2 = "I enjoy eating apples"
similarity = calculate_sentence_similarity(sentence1, sentence2, model)
print(f"相似度 '{sentence1}' 与 '{sentence2}': {similarity:.4f}")

# 测试不相似句子对
sentence3 = "I love eating apples"
sentence4 = "I hate programming computers"
similarity = calculate_sentence_similarity(sentence3, sentence4, model)
print(f"相似度 '{sentence3}' 与 '{sentence4}': {similarity:.4f}")

# 测试更多句子对
sentence5 = "cat sleeping sofa"
sentence6 = "dog running park"
sentence7 = "feline resting couch"
similarity = calculate_sentence_similarity(sentence5, sentence6, model)
print(f"相似度 '{sentence5}' 与 '{sentence6}': {similarity:.4f}")
similarity = calculate_sentence_similarity(sentence5, sentence7, model)
print(f"相似度 '{sentence5}' 与 '{sentence7}': {similarity:.4f}")

def calculate_wmd(sentence1, sentence2, model):
    """
    使用Word Mover's Distance (WMD)计算两个句子之间的距离
    
    Args:
        sentence1 (str): 第一个句子
        sentence2 (str): 第二个句子
        model (fasttext.FastText._FastText): fastText模型
    
    Returns:
        float: 两个句子之间的WMD距离，值越小表示相似度越高
    """
    # 分词
    words1 = sentence1.lower().split()
    words2 = sentence2.lower().split()
    
    # 过滤掉无法获取词向量的单词
    words1 = [word for word in words1 if word.isalpha()]
    words2 = [word for word in words2 if word.isalpha()]
    
    if not words1 or not words2:
        return float('inf')
    
    # 计算词频
    def get_word_freq(words):
        freq = {}
        for word in words:
            freq[word] = freq.get(word, 0) + 1
        return freq
    
    freq1 = get_word_freq(words1)
    freq2 = get_word_freq(words2)
    
    # 计算归一化的词频（作为权重）
    total1 = sum(freq1.values())
    weights1 = {word: count/total1 for word, count in freq1.items()}
    
    total2 = sum(freq2.values())
    weights2 = {word: count/total2 for word, count in freq2.items()}
    
    # 使用numpy实现简单的WMD计算
    # 1. 构建词向量矩阵
    vocab1 = list(weights1.keys())
    vocab2 = list(weights2.keys())
    
    # 2. 计算词向量之间的距离矩阵（余弦距离）
    distance_matrix = np.zeros((len(vocab1), len(vocab2)))
    for i, word1 in enumerate(vocab1):
        vec1 = model.get_word_vector(word1)
        for j, word2 in enumerate(vocab2):
            vec2 = model.get_word_vector(word2)
            # 计算余弦距离
            cosine_similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            cosine_distance = 1 - cosine_similarity
            distance_matrix[i, j] = cosine_distance
    
    # 3. 简单的WMD计算（使用最小距离匹配）
    # 注意：这是一个简化版本，不是真正的最优运输解
    wmd_distance = 0.0
    
    # 调试信息：记录每个词的最接近词
    debug_info = {
        'sentence1_to_sentence2': [],
        'sentence2_to_sentence1': []
    }
    
    # 对于第一个句子中的每个单词
    for i, word1 in enumerate(vocab1):
        weight1 = weights1[word1]
        # 找到第二个句子中距离最近的单词
        min_distance = min(distance_matrix[i])
        min_index = np.argmin(distance_matrix[i])
        closest_word = vocab2[min_index]
        wmd_distance += weight1 * min_distance
        
        # 记录调试信息
        debug_info['sentence1_to_sentence2'].append({
            'word': word1,
            'closest_word': closest_word,
            'distance': min_distance,
            'weight': weight1
        })
    
    # 对于第二个句子中的每个单词
    for j, word2 in enumerate(vocab2):
        weight2 = weights2[word2]
        # 找到第一个句子中距离最近的单词
        min_distance = min(distance_matrix[:, j])
        min_index = np.argmin(distance_matrix[:, j])
        closest_word = vocab1[min_index]
        wmd_distance += weight2 * min_distance
        
        # 记录调试信息
        debug_info['sentence2_to_sentence1'].append({
            'word': word2,
            'closest_word': closest_word,
            'distance': min_distance,
            'weight': weight2
        })
    
    # 输出调试信息
    print("\n=== WMD计算===")
    print(f"句子1: {sentence1}")
    print(f"句子2: {sentence2}")
    
    print("\n句子1中的词在句子2中的最接近词：")
    for item in debug_info['sentence1_to_sentence2']:
        print(f"  '{item['word']}' -> '{item['closest_word']}' (距离: {item['distance']:.4f})")
    
    print("\n句子2中的词在句子1中的最接近词：")
    for item in debug_info['sentence2_to_sentence1']:
        print(f"  '{item['word']}' -> '{item['closest_word']}' (距离: {item['distance']:.4f})")
    
    return wmd_distance

# 测试WMD计算
print("\n=== 测试WMD距离 ===")

# 测试相似句子对
wmd_distance = calculate_wmd(sentence1, sentence2, model)
print(f"WMD距离 '{sentence1}' 与 '{sentence2}': {wmd_distance:.4f}")

# 测试不相似句子对
wmd_distance = calculate_wmd(sentence3, sentence4, model)
print(f"WMD距离 '{sentence3}' 与 '{sentence4}': {wmd_distance:.4f}")

# 测试更多句子对
wmd_distance = calculate_wmd(sentence5, sentence6, model)
print(f"WMD距离 '{sentence5}' 与 '{sentence6}': {wmd_distance:.4f}")
wmd_distance = calculate_wmd(sentence5, sentence7, model)
print(f"WMD距离 '{sentence5}' 与 '{sentence7}': {wmd_distance:.4f}")
