import streamlit as st
import os
import httpx
import fitz
import numpy as np
import nanovector
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

st.set_page_config(page_title="工程AI助手", page_icon="🏗️")

# ==========================================
# 🔐 密码保护拦截器 (保持不变)
# ==========================================
def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets.get("APP_PASSWORD", "123456"):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.title("🔒 请先输入访问密码")
        st.text_input("密码", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.title("🔒 请先输入访问密码")
        st.text_input("密码", type="password", on_change=password_entered, key="password")
        st.error("😕 密码错误，请重试")
        return False
    else:
        return True

if not check_password():
    st.stop() 

# ==========================================
# 正式功能
# ==========================================
st.title("🏗️ 工程规范智能问答助手 (云端版)")
st.write("上传规范后，我会在本地进行语义检索，哪怕字面不一样也能找到！")

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
ZHIPU_KEY = os.getenv("ZHIPU_API_KEY")

if not DEEPSEEK_KEY:
    try:
        DEEPSEEK_KEY = st.secrets["DEEPSEEK_API_KEY"]
        ZHIPU_KEY = st.secrets["ZHIPU_API_KEY"]
    except Exception:
        pass

os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
http_client = httpx.Client(trust_env=False)

chat_client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com", http_client=http_client)
embed_client = OpenAI(api_key=ZHIPU_KEY, base_url="https://open.bigmodel.cn/api/paas/v4/", http_client=http_client)

DIMENSION = 2048
if "vector_index" not in st.session_state:
    st.session_state.vector_index = nanovector.Index(dim=DIMENSION, metric="cosine")
if "documents" not in st.session_state:
    st.session_state.documents = []

def get_embedding(text):
    response = embed_client.embeddings.create(model="embedding-3", input=text)
    return response.data[0].embedding

with st.sidebar:
    st.header("📁 上传知识库")
    uploaded_file = st.file_uploader("请选择一个 PDF 或 TXT 文件", type=["pdf", "txt"])
    if st.button("清空知识库"):
        st.session_state.vector_index = nanovector.Index(dim=DIMENSION, metric="cosine")
        st.session_state.documents = []
        st.session_state.pop("db_loaded", None)
        st.rerun()

# ==========================================
# 🔥 核心修改区域：加入异常捕获和状态打印
# ==========================================
if uploaded_file is not None and "db_loaded" not in st.session_state:
    try:
        with st.spinner("正在解析文件..."):
            # 注意：一定要把文件指针放到最前面
            file_bytes = uploaded_file.read() 
            paragraphs = []
            
            if uploaded_file.name.endswith('.txt'):
                text = file_bytes.decode("utf-8")
                paragraphs = text.split("\n")
            elif uploaded_file.name.endswith('.pdf'):
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                for page in doc:
                    text = page.get_text()
                    clean_text = text.replace("\n", "").replace(" ", "")
                    paragraphs.extend(clean_text.split("。"))

            paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 5]
            
            if not paragraphs:
                st.warning("⚠️ 解析出的文本为空。如果您上传的是扫描版 PDF（纯图片），AI 无法读取文字。请换一个文字版 PDF 或 TXT 测试。")
            else:
                st.info(f"成功解析出 {len(paragraphs)} 段文本，正在转换为 AI 向量...")
                
                # 显示进度条
                progress_bar = st.progress(0)
                for i, p in enumerate(paragraphs):
                    vector = get_embedding(p)
                    st.session_state.vector_index.add(f"doc_{i}", np.array(vector, dtype=np.float32), metadata=p)
                    # 更新进度条
                    progress_bar.progress((i + 1) / len(paragraphs))
                
                st.session_state.documents = paragraphs
                st.session_state.db_loaded = True
                st.sidebar.success(f"✅ 已成功索引 {len(paragraphs)} 条规范！")

    except Exception as e:
        # 🔥 只要有报错，直接显示在网页上，绝不静默卡死
        st.error(f"❌ 处理文件时发生错误：{e}")

# 聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 核心交互
if user_question := st.chat_input("像平时说话一样提问（例如：如果工地起火了，应该注意什么？）"):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        if "db_loaded" not in st.session_state:
            st.warning("请先在左侧上传规范文件！")
        else:
            with st.spinner("正在本地检索相似内容..."):
                try:
                    question_vector = np.array(get_embedding(user_question), dtype=np.float32)
                    results = st.session_state.vector_index.search(question_vector, top_k=3)
                    relevant_text = "\n".join([r.metadata for r in results])

                    prompt = f"""你是一个资深总监理工程师。请严格根据以下参考资料回答问题。
如果资料中没有答案，请直接回答“规范未提及”，不要自己编造。
回答时，请尽量引用原文中的关键数字或条款。

【参考资料】：
{relevant_text}

【问题】：{user_question}"""

                    response = chat_client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    answer = response.choices[0].message.content
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                    with st.expander("🔍 点击查看本次回答参考的规范原文"):
                        for idx, res in enumerate(results):
                            st.info(f"**参考片段 {idx + 1}：**\n{res.metadata}")
                except Exception as e:
                    st.error(f"❌ AI 回答失败：{e}")
