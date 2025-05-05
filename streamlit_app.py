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

st.set_page_config(page_title="きいてミライ（β）", layout="wide", page_icon="🏛️")

top_k = 3
GPT_MODEL = "gpt-4.1-mini"
GPT_TEMPERATURE = 0.1

for key in ["agreed", "query", "send_now", "last_answer", "last_matches", "is_generating", "input", "input_value", "clarified", "clarify_active", "suggestions_sampled"]:
    if key not in st.session_state:
        if key in ["query", "last_answer", "input", "input_value"]:
            st.session_state[key] = ""
        elif key == "suggestions_sampled":
            st.session_state[key] = []
        else:
            st.session_state[key] = False

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
        st.rerun()
    st.stop()

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

def load_faiss_index_and_meta():
    folder_id = st.secrets["kiite-mirai"]["GOOGLE_MIRAI_DATA_ID"]
    creds = Credentials.from_service_account_info(st.secrets["gsheets_service_account"], scopes=["https://www.googleapis.com/auth/drive"])
    service = build("drive", "v3", credentials=creds)
    def download(file_id):
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        MediaIoBaseDownload(fh, request).next_chunk()
        fh.seek(0)
        return fh.read()
    index, meta = None, []
    response = service.files().list(q=f"'{folder_id}' in parents and trashed = false", fields="files(id, name)").execute()
    files = {f["name"]: f["id"] for f in response["files"]}
    if "mirai.index" in files and "mirai.meta.json" in files:
        index = faiss.read_index(io.BytesIO(download(files["mirai.index"])))
        meta = json.loads(download(files["mirai.meta.json"]).decode("utf-8"))
    return index, meta

def search_faiss_and_respond(query):
    index, meta = load_faiss_index_and_meta()
    if index is None or index.ntotal == 0:
        return {
            "matches": [],
            "summary": "🔍 関連する市長の発言は見つかりませんでした。"
        }

    # ベクトル化
    vec = client.embeddings.create(model="text-embedding-ada-002", input=[query])
    qvec = np.array(vec.data[0].embedding, dtype="float32").reshape(1, -1)

    # 検索
    D, I = index.search(qvec, top_k)
    matches = [meta[i] | {"score": float(D[0][j])} for j, i in enumerate(I[0]) if i >= 0]

    # 全体要約生成
    try:
        combined_text = "\n\n".join(m["text"] for m in matches)
        overall_prompt = load_prompt("mirai_summary.txt")
        messages = [
            {"role": "system", "content": overall_prompt},
            {"role": "user", "content": combined_text}
        ]
        resp = client.chat.completions.create(model=GPT_MODEL, messages=messages,temperature = GPT_TEMPERATURE)
        summary = resp.choices[0].message.content.strip()
    except Exception as e:
        summary = f"⚠️ 全体サマリ生成失敗：{e}"

    return {
        "matches": matches,
        "summary": summary
    }



st.title("🏛️ きいてミライ（β）")

# --- チャット欄（送信ボタンなし・Enter送信） ---
st.markdown("---\n\n#### 💬 質問してみよう")
st.text_input(
    label="",
    key="input",
    value=st.session_state.input_value,
    placeholder="例：インバウンド観光に対する取り組みは？",
    on_change=lambda: st.session_state.update(send_now=True)
)


# --- サジェスト ---
if not st.session_state.get("clarify_active", False):  
    suggestions_master = [
        "防災に関する市長の発言はありますか？",
        "子育て支援の方針について教えて",
        "地域活性化に向けた取り組みは？"
    ]
    if not st.session_state.suggestions_sampled:
        st.session_state.suggestions_sampled = random.sample(suggestions_master, k=3)

    cols = st.columns(3)
    for i, s in enumerate(st.session_state.suggestions_sampled):
        if cols[i].button(s, key=f"sugg_{s}"):
            st.session_state.input_value = s
            st.session_state.query = s
            st.session_state.send_now = True
            st.session_state.is_generating = True
            st.session_state.clarify_active = False
            with st.spinner(f"⏳ 「{s}」に回答中... 少々お待ちください"):
                results = search_faiss_and_respond(s)
                st.session_state.last_answer = results["summary"]
                st.session_state.last_matches = results["matches"]
                
                # ログ記録
                try:
                    log_to_gsheet(s, results["summary"])
                except Exception as e:
                    st.warning(f"⚠️ ログ記録に失敗しました: {e}")
                    
            st.session_state.is_generating = False
            st.rerun()

# Clarifyプロンプトの確認
if st.session_state.input and not st.session_state.get("clarified", False):
    clarify_result = clarify_query(st.session_state.input)

    if clarify_result["ambiguous"] and clarify_result["rewritten_query"]:
        st.session_state.clarify_active = True
        st.info(f"👇 より正確な検索のため「{clarify_result['reason']}」ことをお勧めします。例えば、以下の質問に置き換えるのはいかがでしょうか？\n\n**→ {clarify_result['rewritten_query']}**")

        col1, col2 = st.columns(2)
        if col1.button("🔁 置き換えて検索"):
            st.session_state.input_value = clarify_result["rewritten_query"]
            st.session_state.clarified = True
            st.session_state.clarify_active = False
            st.session_state.send_now = True
            st.rerun()
        if col2.button("🔜 入力文のままで検索"):
            st.session_state.clarified = True
            st.session_state.clarify_active = False
            st.session_state.send_now = True
            st.rerun()
        st.stop()
    
    else:
        # 🟨 あいまいでない or 明確な修正提案がない場合も、Clarifyは終了させる！
        st.session_state.clarified = True

# --- 送信処理（Enter or サジェスト選択時） ---
if st.session_state.input and st.session_state.send_now:
    st.session_state.send_now = False
    st.session_state.is_generating = True
    with st.spinner(f"⏳ 「{st.session_state.input}」に回答中... 少々お待ちください"):
        results = search_faiss_and_respond(st.session_state.input)
        st.session_state.last_answer = results["summary"]
        st.session_state.last_matches = results["matches"]
        
        # ログ記録
        try:
            log_to_gsheet(st.session_state.input, results["summary"])
        except Exception as e:
            st.warning(f"⚠️ ログ記録に失敗しました: {e}")
            
    st.session_state.input_value = ""
    st.session_state.is_generating = False


st.markdown("#### 💡 要約まとめ")
if st.session_state.is_generating:
    st.info("⏳ 回答中... 少々お待ちください")
elif st.session_state.last_answer:
    st.success(st.session_state.last_answer)
    
    #st.markdown("---\n\n#### 📂 詳細内容")
    #for m in st.session_state.last_matches:
    #    with st.expander(f" {m.get('topic', '未分類')}（{m.get('source_file', '')}）"):
    #        st.markdown(m["text"])

st.divider()
st.caption("""
⚠️ 回答は生成AIによるものであり、正確性を保証するものではありません。  
🙌 本プロジェクトは個人により運営されています。ご支援いただける方はぜひこちらから：  
[💛 codocで支援する](https://codoc.jp/sites/p8cEFlTZQA/entries/MMZnODc1dw)
""")
