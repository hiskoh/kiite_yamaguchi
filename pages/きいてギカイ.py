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
for key in ["agreed", "query", "send_now", "last_answer", "last_matches", "is_generating", "input", "input_value", "suggestions_sampled", "qa_pairs"]:
    if key not in st.session_state:
        if key in ["query", "last_answer", "input", "input_value"]:
            st.session_state[key] = ""
        elif key == "suggestions_sampled":
            st.session_state[key] = []
        elif key in ["last_matches", "qa_pairs"]:
            st.session_state[key] = []
        else:
            st.session_state[key] = False

# ✅ プロンプト読み込み関数
def load_prompt(filename):
    with open("prompts/" + filename, "r", encoding="utf-8") as f:
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
    import tempfile

    gdrive_folder_id = st.secrets["kiite-gikai"]["GOOGLE_GIKAI_DATA_ID"]
    creds = Credentials.from_service_account_info(
        st.secrets["gsheets_service_account"],
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    service = build("drive", "v3", credentials=creds)
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    def list_index_meta_files(folder_id):
        query = f"'{folder_id}' in parents and trashed = false"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        return [f for f in results.get("files", []) if f["name"].endswith(('.index', '.meta.json'))]

    def download_file_content(file_id):
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        return fh.read()

    # 🔹 index/meta をマッピング
    index_files = list_index_meta_files(gdrive_folder_id)
    file_pairs = {}
    for f in index_files:
        if f["name"].endswith(".meta.json"):
            base = f["name"].removesuffix(".meta.json")
            file_pairs.setdefault(base, {})["meta_id"] = f["id"]
        elif f["name"].endswith(".index"):
            base = f["name"].removesuffix(".index")
            file_pairs.setdefault(base, {})["index_id"] = f["id"]

    # 🔹 クエリベクトル化
    res = client.embeddings.create(model="text-embedding-ada-002", input=[query])
    query_vec = np.array(res.data[0].embedding, dtype="float32").reshape(1, -1)

    matches = []
    meta_by_file = {}

    # 🔍 類似チャンク検索
    for base, pair in file_pairs.items():
        if "index_id" not in pair or "meta_id" not in pair:
            continue

        meta = json.loads(download_file_content(pair["meta_id"]))
        for m in meta:
            m["source_file"] = base + ".txt"
        meta_by_file[base + ".txt"] = meta

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(download_file_content(pair["index_id"]))
            index = faiss.read_index(f.name)

        if index.ntotal == 0 or index.d != query_vec.shape[1]:
            continue

        D, I = index.search(query_vec, top_k)
        for i, dist in zip(I[0], D[0]):
            if i == -1 or i >= len(meta):
                continue
            m = meta[i]
            m["score"] = float(dist)
            m["source_index"] = base
            matches.append(m)

    if not matches:
        return {
            "matches": [],
            "summary": "🔍 関連する議事録の抜粋は見つかりませんでした。",
            "qa_pairs": []
        }

    # 🔹 スコア上位抽出
    top_matches = sorted(matches, key=lambda x: x["score"])[:top_k]

    # 🔹 Q/Aペア抽出（補完含む）
    pair_matches = []
    seen = set()
    for m in top_matches:
        pid = m.get("pair_id")
        src = m.get("source_file")
        if not pid or src not in meta_by_file or (src, pid) in seen:
            continue
        seen.add((src, pid))
        group = [x for x in meta_by_file[src] if x.get("pair_id") == pid]
        q = [x for x in group if x.get("qa_role") == "Q"]
        a = [x for x in group if x.get("qa_role") == "A"]
        pair_matches.append({"pair_id": pid, "source_file": src, "Q": q, "A": a})

    # ✅ プロンプト読み込み（ファイル名は文字列）
    gikai_pair_prompt = load_prompt("gikai_pair_summary.txt")
    summary_overall_prompt = load_prompt("gikai_summary_overall.txt")

    summary_overall = "⚠️ 情報が見つかりませんでした。"
    # ✅ Q/Aペアの個別要約
    summary_per_pair = []
    for pair in pair_matches:
        q_texts = [q["text"] for q in pair["Q"]]
        a_texts = [a["text"] for a in pair["A"]]
        qa_context = "\n\n".join(["【質問】" + q for q in q_texts] + ["【答弁】" + a for a in a_texts])
        try:
            messages = [
                {"role": "system", "content": gikai_pair_prompt},
                {"role": "user", "content": qa_context}
            ]
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            summary = resp.choices[0].message.content.strip()
        except Exception as e:
            summary = f"⚠️ 要約失敗：{e}"
        pair["summary"] = summary
        summary_per_pair.append(summary)

    # ✅ 全体要約
    if summary_per_pair:
        try:
            context = "\n\n".join([f"{i+1}件目：{s}" for i, s in enumerate(summary_per_pair)])
            messages = [
                {"role": "system", "content": summary_overall_prompt},
                {"role": "user", "content": context}
            ]
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            summary_overall  = resp.choices[0].message.content.strip()
        except Exception as e:
            summary_overall = f"⚠️ 全体サマリ生成失敗：{e}"
    else:
        summary_overall = "⚠️ 情報が見つかりませんでした。"

    return {
        "matches": top_matches,
        "summary": summary_overall,
        "qa_pairs": pair_matches
    }




# 🔸 UI構成
st.title("📜 きいてギカイやまぐち（β）")

# --- キャラクターとサジェスト ---
#st.image("character.gif", width=100)


# --- チャット欄（送信ボタンなし・Enter送信） ---
st.markdown("---\n\n#### 💬 質問してみよう")
st.text_input(
    label="",
    key="input",
    value=st.session_state.input_value,
    placeholder="例：物価高が心配…",
    on_change=lambda: st.session_state.update(send_now=True)
)

# --- サジェスト ---
suggestions_master = [
    "子育て政策について不安です...",
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
            st.session_state.qa_pairs = results["qa_pairs"]
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
        st.session_state.qa_pairs = results["qa_pairs"]
    st.session_state.input_value = ""
    st.session_state.is_generating = False


# --- 回答欄 ---
st.markdown("#### 💡議会質問のまとめ")

if st.session_state.is_generating:
    st.info("⏳ 回答中... 少々お待ちください")
elif st.session_state.last_answer and st.session_state.qa_pairs:
    st.success(st.session_state.last_answer)

    st.markdown("---\n\n#### 📂 各質問の要約と原文")
    for i, pair in enumerate(st.session_state.qa_pairs, start=1):
        summary = pair.get("summary", "").strip()
        if not summary:
            continue

        st.markdown(f"##### {i}. {summary}")
        
        for q in pair.get("Q", []):
            with st.expander(f"🟢【質問】{q.get('speaker_role')} {q.get('speaker')}（{q.get('source_file')}）"):
                st.markdown(q.get("text", ""))

        for a in pair.get("A", []):
            with st.expander(f"🔵【答弁】{a.get('speaker_role')} {a.get('speaker')}（{a.get('source_file')}）"):
                st.markdown(a.get("text", ""))
elif st.session_state.send_now or st.session_state.input.strip() or st.session_state.query:
    st.warning("⚠️ 情報が見つかりませんでした。")

# --- フッター 
st.divider() 

st.caption("""
⚠️ 回答は生成AIによるものであり、正確性を保証するものではありません。  
🙌 本プロジェクトは個人により運営されています。ご支援いただける方はぜひこちらから：  
[💛 codocで支援する](https://codoc.jp/sites/p8cEFlTZQA/entries/MMZnODc1dw)
""")
