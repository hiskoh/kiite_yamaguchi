import streamlit as st
from openai import OpenAI
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from datetime import datetime
import random
# ✅ ページ設定
import streamlit as st

folder_id = st.secrets["kiite-gikai"]["GOOGLE_DRIVE_FOLDER_ID"]
st.write("🔍 Using folder ID:", folder_id)

creds = Credentials.from_service_account_info(
    st.secrets["gsheets_service_account"],
    scopes=["https://www.googleapis.com/auth/drive"]
)
service = build("drive", "v3", credentials=creds)

try:
    result = service.files().list(q=f"'{folder_id}' in parents", pageSize=1).execute()
    st.success("✅ Drive access succeeded.")
except Exception as e:
    st.error(f"❌ Drive access failed: {e}")


st.set_page_config(page_title="きいてギカイやまぐち（β）", layout="wide", page_icon="📜")

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
    st.title("📜きいてギカイやまぐち（β）")

    st.markdown("""
    ### ご利用にあたってのご案内

    - このチャットでは、山口市議会の議事録をもとに、議会でどんな議論が行われているかを知ることができます。  
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
    sheet = client_gs.open_by_key(st.secrets["kiite-gikai"]["GOOGLE_GIKAI_LOG_SHEET_ID"]).worksheet("logs")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sheet.append_row([now, question, answer])

# Google Drive から .txt ファイルを取得
def load_gikai_data():
    creds = Credentials.from_service_account_info(
        st.secrets["gsheets_service_account"],
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    service = build("drive", "v3", credentials=creds)

    folder_id = st.secrets["kiite-gikai"]["GOOGLE_DRIVE_FOLDER_ID"]
    files = list_txt_files_recursive(service, folder_id)

    combined_text = ""
    for f in files:
        content = download_file_content(service, f["id"])
        combined_text += f"\n\n【{f['name']}】\n{content}"
    
    return combined_text.strip()
    
def list_txt_files_recursive(service, folder_id):
    query = f"'{folder_id}' in parents and trashed = false"
    files = []
    page_token = None

    while True:
        response = service.files().list(
            q=query,
            spaces='drive',
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token
        ).execute()
        
        for file in response.get('files', []):
            if file['mimeType'] == 'application/vnd.google-apps.folder':
                # サブフォルダを再帰探索
                files.extend(list_txt_files_recursive(service, file['id']))
            elif file['name'].endswith(".txt"):
                files.append(file)

        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break

    return files

def download_file_content(service, file_id):
    file = service.files().get_media(fileId=file_id).execute()
    return file.decode("utf-8")


# ✅ Enter送信処理（テキスト確定時）
def on_enter():
    if st.session_state.input.strip():
        st.session_state.send_now = True

def ask_and_display_answer(user_query):
    st.session_state.query = user_query
    st.session_state.is_generating = True
    with st.spinner(f"⏳ 「{user_query}」に回答中... 少々お待ちください"):
        gikai_context = load_gikai_data()
        system_prompt = f"{load_prompt()}\n\n{gikai_context}"
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
st.title("📜 きいてギカイやまぐち（β）")

# --- キャラクターとサジェスト ---
#st.image("character.gif", width=100)

st.write("") 

# --- チャット欄（送信ボタンなし・Enter送信） ---
st.markdown("#### 💬 質問してみよう")
st.text_input(
    label="",
    key="input",
    value=st.session_state.input_value,
    placeholder="例：物価高が心配…",
    on_change=lambda: st.session_state.update(send_now=True)
)

# --- サジェスト ---
suggestions_master = [
    "子育てが不安...",
    "学校教育について気になる",
    "行政のDXって何か話題になっている？"
]
if not st.session_state.suggestions_sampled:
    st.session_state.suggestions_sampled = random.sample(suggestions_master, k=3)

cols = st.columns(3)
for i, s in enumerate(st.session_state.suggestions_sampled):
    if cols[i].button(f" {s}", key=f"sugg_{s}"):
        st.session_state.input_value = s
        st.session_state.query = s
        st.session_state.send_now = False   # ← 強制的に送信を止める
        ask_and_display_answer(s)           # 即時回答（UX重視）
        st.rerun()                          # 再描画して検索窓に反映

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
