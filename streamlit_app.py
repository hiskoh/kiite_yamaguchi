import streamlit as st
from openai import OpenAI
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# 🔑 OpenAI クライアント
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 🔐 Google Sheets 接続
def get_gspread_client():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gsheets_service_account"], scope)
    return gspread.authorize(creds)

# 📝 ログ記録用
def log_to_gsheet(question, answer):
    client_gs = get_gspread_client()
    sheet = client_gs.open_by_key(st.secrets["GOOGLE_LOG_SHEET_ID"]).worksheet("logs")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sheet.append_row([now, question, answer])

# 📚 山口市の情報をdataシートから読み込む
def load_yamaguchi_data():
    client_gs = get_gspread_client()
    sheet = client_gs.open_by_key(st.secrets["GOOGLE_DATA_SHEET_ID"]).worksheet("data")
    rows = sheet.get_all_records()
    combined_info = ""
    for row in rows:
        combined_info += f"\n\n【{row['カテゴリ']}】{row['タイトル']}：{row['本文']}"
    return combined_info.strip()

# 🌞 UI設定
st.set_page_config(page_title="聞いてみらい山口", page_icon="🌞")
st.title("🌞聞いてみらい山口")
st.write("山口市の“これから”を、一緒に考えるチャットです。気になることを、気軽に聞いてみてください。")

query = st.text_input("気になることを入力してください")

# 🤖 回答生成
if query:
    with st.spinner("回答を生成中..."):
        yamaguchi_context = load_yamaguchi_data()

        system_prompt = f"""
あなたは山口市の行政文書や政策に詳しい親切なアシスタントです。
以下の情報は山口市が公開している計画・政策の一部です。
必要に応じて内容を参考にして、市民にわかりやすく、丁寧に答えてください。
以下に載っていない情報や政治的な主張、山口市の行政とは関係ない質問に関しては一貫して「申し訳ありません、その質問にはお答えできません」と回答してください。

{yamaguchi_context}
"""

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ]
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"⚠️ エラーが発生しました：{e}"

    log_to_gsheet(query, answer)
    st.write("🤖 **聞いてみらい山口の回答**")
    st.success(answer)

# 🔻 注意書き・支援リンク
st.caption("""
📌 本チャットの内容は、みなさんの関心や疑問をもとに、よくある質問を整理したり、行政との新しいコミュニケーションの形をつくっていくことを目的に記録させていただいています。個人情報は入力しないようお願いいたします。また、内容の記録に同意された方のみ、チャット入力・送信をお願いします。
⚠️ 回答は生成AIによるものであり、正確性を保証するものではありません。  
🙌 本プロジェクトは個人により運営されています。ご支援いただける方はぜひこちらから：  
[💛 codocで支援する](https://codoc.jp/sites/p8cEFlTZQA/entries/MMZnODc1dw)
""")
