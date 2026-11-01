import os
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

import httpx
from openai import OpenAI

# 1. 读取文件并切分
file_path = r"C:\Users\Administrator\Desktop\我的工程AI知识库\guifan.txt"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()
paragraphs = content.split("\n")

# 2. 配置 AI
http_client = httpx.Client(trust_env=False)
client = OpenAI(
    api_key="sk-1b9e4c9dc61a4ff785e513cce2a98b95", 
    base_url="https://api.deepseek.com",
    http_client=http_client
)

print("=== 工程AI助手已启动，输入'退出'结束对话 ===")

# 3. 开始无限循环
while True:
    keyword = input("\n请输入你要查的关键词：")
    
    # 退出机制
    if keyword == "退出":
        print("程序结束。")
        break
        
    relevant_text = ""
    for p in paragraphs:
        if keyword in p:
            relevant_text = relevant_text + p + "\n"

    if relevant_text == "":
        print("没找到相关资料，无法回答。")
    else:
        print("已找到规范内容，正在呼叫 AI 专家...\n")
        
        prompt = f"请根据以下参考资料，用专业且通俗的语言，总结一下关于【{keyword}】的要求。\n\n【参考资料】：\n{relevant_text}"

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        print("=========== AI 的总结 ===========")
        print(response.choices[0].message.content)
        print("=================================")