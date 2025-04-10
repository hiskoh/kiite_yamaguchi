import streamlit as st

st.set_page_config(page_title="聞いてみらい山口", page_icon="🌞")

st.title("🌞聞いてみらい山口")
st.write("山口市の“これから”を、一緒に考えるチャットです。")

query = st.text_input("気になることを入力してください")

if query:
    # ここに生成AIの呼び出しなどを実装予定
    st.write(f"あなたの質問：{query}")
    st.write("👉（ここに回答が返ってきます）")
