import streamlit as st
import json, random, faiss, io
import numpy as np
from datetime import datetime
from openai import OpenAI
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.service_account import Credentials
from oauth2client.service_account import ServiceAccountCredentials
import gspread
import tempfile

st.set_page_config(page_title="きいてミライ（β）", layout="wide", page_icon="🏛️")

top_k = 3
GPT_MODEL = "gpt-4.1-mini"
GPT_TEMPERATURE = 0.1

# セッション状態初期化
for key in ["agreed", "query", "send_now", "last_answer", "last_matches", "is_generating", "input", "input_value", "clarified", "clarify_active", "suggestions_sampled", "mirai_index_cache"]:
    if key not in st.session_state:
        if key in ["query", "last_answer", "input", "input_value"]:
            st.session_state[key] = ""
        elif key == "suggestions_sampled":
            st.session_state[key] = []
        elif key == "mirai_index_cache":
            st.session_state[key] = {}
        else:
            st.session_state[key] = False

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def get_gspread_client():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gsheets_service_account"], scope)
    return gspread.authorize(creds)

def log_to_gsheet(question, answer):
    client_gs = get_gspread_client()
    sheet = client_gs.open_by_key(st.secrets["kiite-mirai"]["GOOGLE_MIRAI_LOG_SHEET_ID"]).worksheet("logs")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sheet.append_row([now, question, answer])

def load_prompt(name):
    with open(f"prompts/{name}", "r", encoding="utf-8") as f:
        return f.read()

def clarify_query(user_query):
    clarify_prompt = load_prompt("mirai_clarify_prompt.txt")
    messages = [
        {"role": "system", "content": clarify_prompt},
        {"role": "user", "content": f"【質問】{user_query}"}
    ]
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            temperature=GPT_TEMPERATURE
        )
        content = response.choices[0].message.content.strip()
        return json.loads(content)
    except Exception as e:
        st.error(f"Clarifyエラー: {e}")
        return {"ambiguous": False, "reason": "", "rewritten_query": ""}

# 同意画面とキャッシュロード
if not st.session_state.agreed:
    st.title("🏛️きいてミライやまぐち（β）")
    st.markdown("""
    ### ご利用にあたってのご案内

    - このチャットでは、山口市長の過去の発言（定例会見、議会での施政方針）をもとに、市長の見解を知ることができます。  
    - **個人情報（氏名・住所・連絡先など）の入力は行わないでください。**  
    - チャット内容は記録されます。内容の記録に同意された方のみ、チャットをご利用ください。
    """)
    st.warning("このチャットを利用するには、以下の内容に同意いただく必要があります。")
    if st.button("✅ 同意してチャットをはじめる"):
        st.session_state.agreed = True

        # 🔹 キャッシュ読み込み開始
        gdrive_folder_id = st.secrets["kiite-mirai"]["GOOGLE_MIRAI_DATA_ID"]
        creds = Credentials.from_service_account_info(
            st.secrets["gsheets_service_account"],
            scopes=["https://www.googleapis.com/auth/drive"]
        )
        service = build("drive", "v3", credentials=creds)

        def list_index_meta_files(folder_id):
            query = f"'{folder_id}' in parents and trashed = false"
            results = service.files().list(q=query, fields="files(id, name)").execute()
            return [f for f in results.get("files", []) if f["name"].endswith(('.index', '.meta.json'))]

        def download_file_content(file_id):
            request = service.files().get_media(fileId=file_id)
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            while True:
                _, done = downloader.next_chunk()
                if done: break
            fh.seek(0)
            return fh.read()

        file_pairs = {}
        for f in list_index_meta_files(gdrive_folder_id):
            base = f["name"].removesuffix(".index").removesuffix(".meta.json")
            file_pairs.setdefault(base, {})
            if f["name"].endswith(".index"):
                file_pairs[base]["index_id"] = f["id"]
            else:
                file_pairs[base]["meta_id"] = f["id"]

        for base, pair in file_pairs.items():
            if "index_id" in pair and "meta_id" in pair:
                meta = json.loads(download_file_content(pair["meta_id"]).decode("utf-8"))
                with tempfile.NamedTemporaryFile(delete=False) as f:
                    f.write(download_file_content(pair["index_id"]))
                    index = faiss.read_index(f.name)
                st.session_state["mirai_index_cache"][base] = {"meta": meta, "index": index}

        st.rerun()
    st.stop()

# 発言録を検索
def search_faiss_and_respond(query):
    res = client.embeddings.create(model="text-embedding-ada-002", input=[query])
    query_vec = np.array(res.data[0].embedding, dtype="float32").reshape(1, -1)

    matches = []
    for base, pair in st.session_state["mirai_index_cache"].items():
        meta = pair["meta"]
        index = pair["index"]
        if index.ntotal == 0 or index.d != query_vec.shape[1]:
            continue
        D, I = index.search(query_vec, top_k)
        for i, dist in zip(I[0], D[0]):
            if i == -1 or i >= len(meta): continue
            m = meta[i]
            m["score"] = float(dist)
            m["source_index"] = base
            matches.append(m)

    if not matches:
        return {
            "matches": [],
            "summary": "🔍 関連する市長の発言は見つかりませんでした。"
        }

    top_matches = sorted(matches, key=lambda x: x["score"])[:top_k]
    try:
        combined_text = "\n\n".join(m["text"] for m in top_matches)
        summary_prompt = load_prompt("mirai_summary.txt")
        messages = [
            {"role": "system", "content": summary_prompt},
            {"role": "user", "content": combined_text}
        ]
        resp = client.chat.completions.create(model=GPT_MODEL, messages=messages, temperature=GPT_TEMPERATURE)
        summary = resp.choices[0].message.content.strip()
    except Exception as e:
        summary = f"⚠️ 全体サマリ生成失敗：{e}"

    return {
        "matches": top_matches,
        "summary": summary
    }
