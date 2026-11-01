import streamlit as st
import os
import httpx
import fitz
import numpy as np
import nanovector  # 零依赖纯 Python 向量库
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# 1. 加载 Keys
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
ZHIPU_KEY = os.getenv("ZHIPU_API_KEY")

# 2. 客户端配置
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
http_client = httpx.Client(trust_env=False)

chat_client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com", http_client=http_client)
embed_client = OpenAI(api_key=ZHIPU_KEY, base_url="https://open.bigmodel.cn/api/paas/v4/", http_client=http_client)

st.set_page_config(page_title="工程AI助手", page_icon="🏗️")
st.title("🏗️ 工程规范智能问答助手 (语义版)")
st.write("上传规范后，我会在本地进行语义检索，哪怕字面不一样也能找到！")

# 3. 初始化 nanovector 索引
# 智谱 embedding-3 的向量维度是 2048
DIMENSION = 2048
index = nanovector.Index(dim=DIMENSION, metric="cosine")

# 4. 把文本变成向量的函数
def get_embedding(text):
    response = embed_client.embeddings.create(
        model="embedding-3",
        input=text
    )
    return response.data[0].embedding

with st.sidebar:
    st.header("📁 上传知识库")
    uploaded_file = st.file_uploader("请选择一个 PDF 或 TXT 文件", type=["pdf", "txt"])
    if st.button("清空聊天记录"):
        st.session_state.messages = []
        # 重置索引
        index = nanovector.Index(dim=DIMENSION, metric="cosine")
        st.session_state.pop("db_loaded", None)
        st.rerun()

# 5. 解析文件并构建本地索引
if uploaded_file is not None and "db_loaded" not in st.session_state:
    paragraphs = []
    if uploaded_file.name.endswith('.txt'):
        text = uploaded_file.read().decode("utf-8")
        paragraphs = text.split("\n")
    elif uploaded_file.name.endswith('.pdf'):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in doc:
            text = page.get_text()
            clean_text = text.replace("\n", "").replace(" ", "")
            paragraphs.extend(clean_text.split("。"))
            
    paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 5]
    
    with st.spinner("正在将规范存入 AI 大脑（向量化）..."):
        for i, p in enumerate(paragraphs):
            vector = get_embedding(p)
            # 将向量和原文一起存入 nanovector
            index.add(f"doc_{i}", np.array(vector, dtype=np.float32), metadata=p)
        
        st.session_state.db_loaded = True
        st.sidebar.success(f"已成功索引 {len(paragraphs)} 条规范！")

# 6. 聊天记录逻辑
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 7. 核心交互
if user_question := st.chat_input("像平时说话一样提问（例如：如果工地起火了，应该注意什么？）"):
    
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)
    
    with st.chat_message("assistant"):
        if "db_loaded" not in st.session_state:
            st.warning("请先在左侧上传规范文件！")
        else:
            with st.spinner("正在本地检索相似内容..."):
                # A. 把用户的问题变成向量
                question_vector = np.array(get_embedding(user_question), dtype=np.float32)
                
                # B. 在 nanovector 中搜索最相似的 3 个段落
                results = index.search(question_vector, top_k=3)
                
                # C. 提取检索到的文本
                relevant_text = "\n".join([r.metadata for r in results])
                
                # D. 发给 DeepSeek 生成最终回答
                prompt = f"你是一个资深总监理工程师。请根据以下参考资料回答问题。\n\n【参考资料】：\n{relevant_text}\n\n【问题】：{user_question}"

                response = chat_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}]
                )
                
                answer = response.choices[0].message.content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})python -m streamlit run web_app.py
               