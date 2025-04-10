import streamlit as st
import openai
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# 🌞 ページ設定
st.set_page_config(page_title="聞いてみらい山口", page_icon="🌞")

# 🔐 APIキー設定（Streamlit Secrets から取得）
openai.api_key = st.secrets["OPENAI_API_KEY"]

# 📊 Google Sheets にログを記録する関数
def log_to_gsheet(question, answer):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(
        st.secrets["gsheets_service_account"], scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key("1mTB1SwNNst80HjdDAFVB1PyN5HuaoMzKzTpKFhu2Xqw").worksheet("logs")  # スプレッドシートID・シート名を合わせておく
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sheet.append_row([now, question, answer])

# 🖼️ タイトルと説明
st.title("🌞聞いてみらい山口")
st.write("山口市の“これから”を、一緒に考えるチャットです。気になることを、気軽に聞いてみてください。")

# 📝 ユーザーの質問入力
query = st.text_input("気になることを入力してください")

# 🤖 ChatGPT（gpt-3.5-turbo）で回答生成
if query:
    with st.spinner("回答を生成中..."):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "あなたは山口市の行政文書や政策に詳しい親切なアシスタントです。市民にわかりやすく、丁寧に答えてください。"},
                    {"role": "user", "content": query}
                ]
            )
            answer = response.choices[0].message["content"].strip()
        except Exception as e:
            answer = f"⚠️ エラーが発生しました：{e}"

    # 💾 回答をスプレッドシートにログとして記録
    log_to_gsheet(query, answer)

    # 📣 結果を表示
    st.write("🗨️ **あなたの質問**")
    st.info(query)

    st.write("🤖 **聞いてみらい山口の回答**")
    st.success(answer)
