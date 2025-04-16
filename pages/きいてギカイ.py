import streamlit as st
import chardet
from openai import OpenAI
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from datetime import datetime
import random, json, io, faiss, tempfile
import numpy as np
from googleapiclient.http import MediaIoBaseDownload
# ✅ ページ設定
import streamlit as st

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
    file_data = service.files().get_media(fileId=file_id).execute()
    detected = chardet.detect(file_data)
    encoding = detected["encoding"] or "utf-8"
    return file_data.decode(encoding, errors="replace")


# ✅ Enter送信処理（テキスト確定時）
def on_enter():
    if st.session_state.input.strip():
        st.session_state.send_now = True

# 議事録データにアクセスして関連発言を出力
def search_faiss_and_respond(query, top_k=5):
    # 議事録データのフォルダID
    gdrive_folder_id = st.secrets["kiite-gikai"]["GOOGLE_GIKAI_DATA_ID"]
    
    # 🔐 Google Drive認証
    creds = Credentials.from_service_account_info(
        st.secrets["gsheets_service_account"], scopes=["https://www.googleapis.com/auth/drive"]
    )
    service = build("drive", "v3", credentials=creds)

    # 🔽 Driveから .index / .meta.json をダウンロード
    def list_index_meta_files(folder_id):
        query = f"'{folder_id}' in parents and trashed = false"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get("files", [])
        return [f for f in files if f["name"].endswith(('.index', '.meta.json'))]

    def download_file_content(file_id):
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        return fh.read()

    index_files = list_index_meta_files(gdrive_folder_id)
    file_pairs = {}
    for f in index_files:
        base = f["name"].rsplit(".", 1)[0]
        file_pairs.setdefault(base, {})
        if f["name"].endswith(".index"):
            file_pairs[base]["index_id"] = f["id"]
        elif f["name"].endswith(".meta.json"):
            file_pairs[base]["meta_id"] = f["id"]

    # ✅ OpenAIインスタンス
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    embedding = client.embeddings.create(model="text-embedding-ada-002", input=[query]).data[0].embedding
    query_vec = np.array(embedding).astype("float32").reshape(1, -1)

    matches = []
    for base, pair in file_pairs.items():
        if "index_id" not in pair or "meta_id" not in pair:
            continue
        # temp保存してロード
        with tempfile.NamedTemporaryFile(delete=False) as index_file:
            index_file.write(download_file_content(pair["index_id"]))
            index_path = index_file.name
        with tempfile.NamedTemporaryFile(delete=False, mode="w+b") as meta_file:
            meta_file.write(download_file_content(pair["meta_id"]))
            meta_path = meta_file.name

        index = faiss.read_index(index_path)
        D, I = index.search(query_vec, top_k)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        for i, dist in zip(I[0], D[0]):
            if i < len(meta):
                m = meta[i]
                m["score"] = float(dist)
                matches.append(m)

    # スコア順に上位抽出
    top_matches = sorted(matches, key=lambda x: x["score"])[:top_k]
    context = "\n\n".join([
        f"{m['speaker_role']} {m['speaker']}（{m['source_file']}）\n{m['text']}"
        for m in top_matches
    ])

    # ✅ GPTによる要約回答
    prompt = f"""以下は議会での発言記録の一部です。
これを参考にして「{query}」という質問に市議会がどう向き合っているか要約してください。

{context}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "あなたは議会の発言を要約するアシスタントです。正確性を重視してください。"},
                {"role": "user", "content": prompt}
            ]
        )
        summary = response.choices[0].message.content.strip()
    except Exception as e:
        summary = f"⚠️ GPTによる要約に失敗しました：{e}"

    return {
        "matches": top_matches,
        "summary": summary
    }


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
        st.session_state.send_now = False
        st.session_state.is_generating = True
        with st.spinner(f"⏳ 「{s}」に回答中... 少々お待ちください"):
            results = search_faiss_and_respond(s, 5)
            st.session_state.last_answer = results["summary"]
            st.session_state.last_matches = results["matches"]
        st.session_state.is_generating = False
        st.rerun()

# --- 送信処理（Enter or サジェスト選択時） ---
if st.session_state.input and st.session_state.send_now:
    st.session_state.send_now = False
    st.session_state.is_generating = True
    with st.spinner(f"⏳ 「{st.session_state.input}」に回答中... 少々お待ちください"):
        results = search_faiss_and_respond(st.session_state.input, 5)
        st.session_state.last_answer = results["summary"]
        st.session_state.last_matches = results["matches"]
    st.session_state.input_value = ""
    st.session_state.is_generating = False


# --- 回答欄 ---
st.markdown("#### 💡議会の発言にもとづく要約")
if st.session_state.is_generating:
    st.info("⏳ 回答中... 少々お待ちください")
elif st.session_state.last_answer:
    st.success(st.session_state.last_answer)
    st.write("マッチ数:", len(top_matches))
    
    # --- 原文チャンク表示（上位類似）
    if st.session_state.last_matches:
        st.markdown("#### 🧾 関連する議事録の抜粋")
        for i, m in enumerate(st.session_state.last_matches, start=1):
            with st.expander(f"{i}. {m['speaker_role']} {m['speaker']}（{m['source_file']}）"):
                st.markdown(m["text"])

# --- フッター 
st.divider() 

st.caption("""
⚠️ 回答は生成AIによるものであり、正確性を保証するものではありません。  
🙌 本プロジェクトは個人により運営されています。ご支援いただける方はぜひこちらから：  
[💛 codocで支援する](https://codoc.jp/sites/p8cEFlTZQA/entries/MMZnODc1dw)
""")
