import streamlit as st
from openai import OpenAI
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def log_to_gsheet(question, answer):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gsheets_service_account"], scope)
    client_gs = gspread.authorize(creds)
    sheet = client_gs.open_by_key("1mTB1SwNNst80HjdDAFVB1PyN5HuaoMzKzTpKFhu2Xqw").worksheet("logs")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sheet.append_row([now, question, answer])

st.set_page_config(page_title="聞いてみらい山口", page_icon="🌞")
st.title("🌞聞いてみらい山口")
st.write("山口市の“これから”を、一緒に考えるチャットです。気になることを、気軽に聞いてみてください。")

query = st.text_input("気になることを入力してください")

if query:
    with st.spinner("回答を生成中..."):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "あなたは山口市の行政文書や政策に詳しい親切なアシスタントです。市民にわかりやすく、丁寧に答えてください。"},
                    {"role": "user", "content": query}
                ]
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"⚠️ エラーが発生しました：{e}"

    log_to_gsheet(query, answer)

    st.write("🗨️ **あなたの質問**")
    st.info(query)
    st.write("🤖 **聞いてみらい山口の回答**")
    st.success(answer)
