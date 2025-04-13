import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
import openai

# ページ設定
st.set_page_config(page_title="聞いてみらい山口", page_icon="🌞")
st.title("🌞 聞いてみらい山口")
st.write("### 質問してみよう")

# OpenAIキー設定
openai.api_key = st.secrets["OPENAI_API_KEY"]

# LLM設定
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding()
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)

# データ読み込み
@st.cache_resource(show_spinner=True)
def load_yamaguchi_data():
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    index = VectorStoreIndex.from_documents(docs)
    return index.as_query_engine()

query_engine = load_yamaguchi_data()

# 回答処理関数
def ask_and_display_answer(user_input):
    if not user_input.strip():
        return
    with st.spinner("山口市の資料から探しています..."):
        response = query_engine.query(user_input)
    st.write("### 💡 回答はこちら")
    st.success(response.response)

# Enterで送信される処理（session_state経由）
def on_enter():
    ask_and_display_answer(st.session_state.input)

# テキスト入力欄（Enterで送信可能）
st.text_input("質問してみよう", key="input", on_change=on_enter)

# サジェストボタン
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("山口市の課題は？"):
        ask_and_display_answer("山口市の課題は？")
with col2:
    if st.button("山口市の人口は？"):
        ask_and_display_answer("山口市の人口は？")
with col3:
    if st.button("市役所の建て替えは？"):
        ask_and_display_answer("市役所の建て替えは？")
