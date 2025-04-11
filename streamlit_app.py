import streamlit as st
from openai import OpenAI
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# ✅ ページ設定（1回のみ）
st.set_page_config(page_title="聞いてみらい山口", page_icon="🌞")

# ✅ セッション管理（同意状況とリロードトリガー）
if "agreed" not in st.session_state:
    st.session_state.agreed = False
if "trigger_rerun" not in st.session_state:
    st.session_state.trigger_rerun = False

# ✅ 同意していない場合は利用規約画面を表示
if not st.session_state.agreed:
    st.title("🌞 聞いてみらい山口 - ご利用にあたって")
    st.warning("このチャットを利用するには、以下の内容に同意いただく必要があります。")

    st.markdown("""
    ### ご利用にあたってのご案内

    - 📌 このチャットは、市民の関心や疑問をもとに、よくある質問を整理・可視化し、  
      行政との新しいコミュニケーションの形をつくっていくことを目的に運営されています。  
    - 🔐 **個人情報（氏名・住所・連絡先など）の入力は行わないでください。**  
    - ✅ チャット内容は記録されます。内容の記録に同意された方のみ、チャットをご利用ください。
    """)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ 同意して利用を開始する"):
            st.session_state.agreed = True
            st.session_state.trigger_rerun = True
    with col2:
        if st.button("🚪 同意しない"):
            st.error("ご利用ありがとうございました。")
            st.stop()

# ✅ 外側で rerun（セッション更新が確実に反映された状態で）
if st.session_state.trigger_rerun:
    st.session_state.trigger_rerun = False
    st.experimental_rerun()

# ✅ 同意済みの場合は通常チャット画面を表示
if st.session_state.agreed:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    def get_gspread_client():
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gsheets_service_account"], scope)
        return gspread.authorize(creds)

    def log_to_gsheet(question, answer):
        client_gs = get_gspread_client()
        sheet = client_gs.open_by_key(st.secrets["GOOGLE_LOG_SHEET_ID"]).worksheet("logs")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([now, question, answer])

    def load_yamaguchi_data():
        client_gs = get_gspread_client()
        sheet = client_gs.open_by_key(st.secrets["GOOGLE_DATA_SHEET_ID"]).worksheet("data")
        rows = sheet.get_all_records()
        combined_info = ""
        for row in rows:
            combined_info += f"\n\n【{row['カテゴリ']}】{row['タイトル']}：{row['本文']}"
        return combined_info.strip()

    st.title("🌞聞いてみらい山口")
    st.write("山口市の“これから”を、一緒に考えるチャットです。気になることを、気軽に聞いてみてください。")

    query = st.text_input("気になることを入力してください")

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
        st.write("🤎 **聞いてみらい山口の回答**")
        st.success(answer)

    st.caption("""
⚠️ 回答は生成AIによるものであり、正確性を保証するものではありません。  
🙌 本プロジェクトは個人により運営されています。ご支援いただける方はぜひこちらから：  
[💛 codocで支援する](https://codoc.jp/sites/p8cEFlTZQA/entries/MMZnODc1dw)
""")
