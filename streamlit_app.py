import streamlit as st
from openai import OpenAI
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import random

# ✅ ページ設定
st.set_page_config(page_title="きいてみらい山口", page_icon="🌞")

# ✅ セッションステートの初期化
for key in ["agreed", "query", "send_now", "last_answer", "is_generating", "input", "input_value", "suggestions_sampled"]:
    if key not in st.session_state:
        if key in ["query", "last_answer", "input", "input_value"]:
            st.session_state[key] = ""
        elif key == "suggestions_sampled":
            st.session_state[key] = []
        else:
            st.session_state[key] = False

# ✅ プロンプト読み込み関数
def load_prompt():
    with open("prompts/system_prompt.txt", "r", encoding="utf-8") as f:
        return f.read()

# ✅ 同意画面
if not st.session_state.agreed:
    st.title("🌞 きいてみらい山口")

    st.markdown("""
    ### ご利用にあたってのご案内

    - このチャットは、市民の関心や疑問をもとに、よくある質問を整理・可視化し、  
      行政との新しいコミュニケーションの形をつくっていくことを目指して運営されています。  
    - **個人情報（氏名・住所・連絡先など）の入力は行わないでください。**  
    - チャット内容は記録されます。内容の記録に同意された方のみ、チャットをご利用ください。
    """)
    
    st.warning("このチャットを利用するには、以下の内容に同意いただく必要があります。")
    
    if st.button("✅ 同意してチャットをはじめる"):
        st.session_state.agreed = True
        st.rerun() 
    st.stop()

# ✅ Chatモード（同意済）
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# 🔧 Google Sheets 接続

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
        combined_info += f"\n\n【{row['カテゴリ']} / {row['サブカテゴリ']}】{row['タイトル']}：{row['説明文']}"
    return combined_info.strip()

# ✅ Enter送信処理（テキスト確定時）
def on_enter():
    if st.session_state.input.strip():
        st.session_state.send_now = True

def ask_and_display_answer(user_query):
    st.session_state.query = user_query
    st.session_state.is_generating = True
    with st.spinner("⏳ 回答中... 少々お待ちください"):
        yamaguchi_context = load_yamaguchi_data()
        system_prompt = f"{load_prompt()}\n\n{yamaguchi_context}"
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ]
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"⚠️ エラーが発生しました：{e}"

    log_to_gsheet(user_query, answer)
    st.session_state.last_answer = answer
    st.session_state.is_generating = False

# 🔸 UI構成
st.title("🌞 きいてみらい山口")

# --- キャラクターとサジェスト ---
#st.image("character.gif", width=100)

st.write("") 

# --- チャット欄（送信ボタンなし・Enter送信） ---
st.markdown("#### 💬 質問してみよう")
st.text_input(
    label="",
    key="input",
    value=st.session_state.input_value,
    placeholder="例：山口市の総合計画について教えて",
    on_change=lambda: st.session_state.update(send_now=True)
)

# --- サジェスト ---
suggestions_master = [
    "山口市の課題は？",
    "市役所の建て替えは？",
    "山口市の人口は？"
]
if not st.session_state.suggestions_sampled:
    st.session_state.suggestions_sampled = random.sample(suggestions_master, k=3)

cols = st.columns(3)
for i, s in enumerate(st.session_state.suggestions_sampled):
    if cols[i].button(f" {s}", key=f"sugg_{s}"):
        st.session_state.input_value = s   # 次回描画で検索窓に表示される
        st.session_state.query = s         # ログ用（必要なら）
        st.session_state.send_now = True   # 次回送信処理をトリガー
        st.rerun()                         # ✅ 今すぐ再描画させる！
        ask_and_display_answer(s)          # 回答を表示させる



# --- 送信処理（Enter or サジェスト選択時） ---
if st.session_state.input and st.session_state.send_now:
    st.session_state.send_now = False
    ask_and_display_answer(st.session_state.input)
    st.session_state.input_value = ""

# --- 回答欄 ---
st.markdown("#### 💡回答はこちら")
if st.session_state.is_generating:
    st.info("⏳ 回答中... 少々お待ちください")
elif st.session_state.last_answer:
    st.success(st.session_state.last_answer)


# --- フッター 
st.divider() 

st.caption("""
⚠️ 回答は生成AIによるものであり、正確性を保証するものではありません。  
🙌 本プロジェクトは個人により運営されています。ご支援いただける方はぜひこちらから：  
[💛 codocで支援する](https://codoc.jp/sites/p8cEFlTZQA/entries/MMZnODc1dw)
""")
