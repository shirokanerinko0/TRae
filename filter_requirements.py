import json

# 文件路径
file_prompt2 = r'd:\OneDrive\graduation_project\TRae\data\kafka\requirements_processed_llm_Pro_deepseek-ai_DeepSeek-V3.2_prompt2.json'
file_prompt4 = r'd:\OneDrive\graduation_project\TRae\data\kafka\requirements_processed_llm_Pro_deepseek-ai_DeepSeek-V3.2_prompt4.json'

# 读取prompt4文件，获取所有req_id
with open(file_prompt4, 'r', encoding='utf-8') as f:
    data_prompt4 = json.load(f)
prompt4_req_ids = {req['req_id'] for req in data_prompt4}

print(f'prompt4中有 {len(prompt4_req_ids)} 个需求')

# 读取prompt2文件
with open(file_prompt2, 'r', encoding='utf-8') as f:
    data_prompt2 = json.load(f)

print(f'prompt2原有 {len(data_prompt2)} 个需求')

# 过滤：只保留在prompt4中也存在的req_id
filtered_data = [req for req in data_prompt2 if req['req_id'] in prompt4_req_ids]

print(f'过滤后保留 {len(filtered_data)} 个需求')

# 保存到原文件
with open(file_prompt2, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, indent=2, ensure_ascii=False)

print(f'已保存到 {file_prompt2}')
