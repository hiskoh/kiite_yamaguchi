import streamlit as st
import chardet
from openai import OpenAI
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from datetime import datetime
import random, json, io, faiss, tempfile, re, boto3, json
import numpy as np
from googleapiclient.http import MediaIoBaseDownload

st.set_page_config(page_title="きいてギカイやまぐち（β）", layout="wide", page_icon="📜")



# ====== ▼ 初期値設定 ========================================================

#検索結果の設定
TOP_N_RETURN   = 10            # 最終返す件数（UIは上位10件でOKならこのまま）
TOPK_CANDIDATES = 10           # S3Vectorsからの仮取得件数（上位候補を多めに持ってくる）
SIM_THRESHOLD  = 0.10          # 類似度のしきい値（0.0〜1.0, 1.0に近いほど類似）

#ChatGPTの設定
GPT_MODEL = "gpt-4.1-mini"
GPT_TEMPERATURE = 0.1
EMBED_MODEL    = "text-embedding-3-small"

#AWSの設定
AWS_REGION     = "us-west-2"
OUTPUT_PREFIX  = "council_chunk_jsonl/"  
AWS_ACCESS_KEY_S = st.secrets["AWS-KEY"]["AWS_ACCESS_KEY"]
AWS_SECRET_KEY_S = st.secrets["AWS-KEY"]["AWS_SECRET_KEY"]
S3_INDEX_ARN   = st.secrets["AWS-KEY"]["VECTOR_INDEX_ARN_COUNCIL"]
DATA_BUCKET_NAME = st.secrets["AWS-KEY"]["DATA_BUCKET_NAME_COUNCIL"]

# ========= ▼ ここからS3 Vectorsでの検索を実装 ===============================

def _to_similarity(distance: float) -> float:
    """S3 Vectorsのdistance(=cosine距離) -> 類似度(1 - distance)"""
    try:
        d = float(distance)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, 1.0 - d))

def _base_from_chunk_id(chunk_id: str) -> str:
    # "somefile_001" → "somefile"
    return re.sub(r"_[0-9]{1,3}$", "", chunk_id)
    
def _fetch_original_chunk_for_search(s3_client, chunk_id: str) -> dict | None:
    """
    ベクトル検索で得た chunk_id から、元の JSONL を S3 から引き、該当行を返す。
    形式は Colab の出力と同じ（speaker/speaker_role/pair_id/qa_role を含む）。
    """
    base = _base_from_chunk_id(chunk_id)
    key  = f"{OUTPUT_PREFIX}{base}.jsonl"
    try:
        body = s3_client.get_object(Bucket=DATA_BUCKET_NAME, Key=key)["Body"].read().decode("utf-8")
    except Exception:
        return None

    for line in body.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("chunk_id") == chunk_id:
            return obj
    return None

def _query_s3vectors(query_text: str, top_k_: int, score_threshold: float):
    """
    S3 Vectors を検索し、[{score, distance, key, source_file, chunk_id, original}] を返す。
    original には JSONL の当該オブジェクト（speaker/qa_role等）が入る。
    """
    # クエリ埋め込み
    oai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    emb = oai.embeddings.create(model=EMBED_MODEL, input=query_text)
    qvec = [float(x) for x in emb.data[0].embedding]

    # クライアント（読み取り専用権限でOK）
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_S,
        aws_secret_access_key=AWS_SECRET_KEY_S,
        region_name=AWS_REGION,
    )
    s3v_client = boto3.client(
        "s3vectors",
        aws_access_key_id=AWS_ACCESS_KEY_S,
        aws_secret_access_key=AWS_SECRET_KEY_S,
        region_name=AWS_REGION,
    )

    # 多めに候補取得 → しきい値でフィルタ
    res = s3v_client.query_vectors(
        indexArn=S3_INDEX_ARN,
        queryVector={"float32": qvec},
        topK=max(TOPK_CANDIDATES, top_k_),
        returnMetadata=True,
        returnDistance=True,
    )
    matches = res.get("vectors", []) or []

    out = []
    for m in matches:
        key      = m.get("key") or m.get("id")
        distance = float(m.get("distance", 0.0))
        score    = _to_similarity(distance)

        if score < score_threshold:
            continue

        md       = m.get("metadata") or {}
        source   = md.get("source_file")
        chunk_id = md.get("chunk_id") or key
        original = _fetch_original_chunk_for_search(s3_client, chunk_id)

        out.append({
            "score": score,
            "distance": distance,
            "key": key,
            "source_file": source,
            "chunk_id": chunk_id,
            "original": original,
        })

    # 類似度が高い順
    out.sort(key=lambda x: x["score"], reverse=True)
    return out
    
# ✅ セッションステートの初期化
for key in ["agreed", "query", "send_now", "last_answer", "last_matches", "is_generating", "input", "input_value", "suggestions_sampled", "qa_pairs", "clarified", "clarify_active"]:
    if key not in st.session_state:
        if key in ["query", "last_answer", "input", "input_value"]:
            st.session_state[key] = ""
        elif key == "suggestions_sampled":
            st.session_state[key] = []
        elif key in ["last_matches", "qa_pairs"]:
            st.session_state[key] = []
        elif key == "clarify_active":
            st.session_state[key] = False
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

# clarify機能（質問があいまいなときのフォロー）
def clarify_query(user_query):
    clarify_prompt = gikai_pair_prompt = load_prompt("gikai_clarify_prompt.txt")
    messages = [
        {"role": "system", "content": clarify_prompt},
        {"role": "user", "content": f"【質問】{user_query}"}
    ]
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            temperature=GPT_TEMPERATURE
        )
        content = response.choices[0].message.content.strip()
        result = json.loads(content)
        return result
    except Exception as e:
        st.error(f"Clarifyエラー: {e}")
        return {"ambiguous": False, "reason": "", "rewritten_query": ""}

# 議事録データにアクセスして関連発言を出力
def search_s3vector_and_respond(query, top_k):
    try:
        hits = _query_s3vectors(
            query_text=query,
            top_k_=TOP_N_RETURN,
            score_threshold=SIM_THRESHOLD,
        )
    except Exception as e:
        return {"matches": [], "summary": f" 検索エラーが発生しました: {e}", "qa_pairs": []}

    if not hits:
        return {"matches": [], "summary": " 関連する議事録の抜粋は見つかりませんでした。", "qa_pairs": []}

    # 上位N件
    top_hits = hits[:TOP_N_RETURN]

    # 表示用整形：QAペア抽出に必要な speaker/qa_role/pair_id を original から取り出す
    top_matches = []
    for h in top_hits:
        o = h.get("original") or {}
        top_matches.append({
            # ★ 従来UI互換：スコア（類似度）やsource/chunk_idはそのまま
            "score": float(h.get("score", 0.0)),
            "source_file": h.get("source_file") or o.get("source_file") or "",
            "chunk_id": h.get("chunk_id"),
            "text": o.get("text", ""),
            "speaker": o.get("speaker"),
            "speaker_role": o.get("speaker_role"),
            "pair_id": o.get("pair_id"),
            "qa_role": o.get("qa_role"),
        })

    # Q/Aペアを組み立て（議員=Q、答弁=A）
    # 🔸 元の「Google Driveメタから復元」ロジックを置換：同ファイルのJSONL全文を再取得して pair_id で束ねる
    # まず、同一 source_file ごとに全行を読んで pair_id -> [chunks] を作る
    from collections import defaultdict
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_S,
        aws_secret_access_key=AWS_SECRET_KEY_S,
        region_name=AWS_REGION,
    )
    def _load_all_for_source(source_file: str):
        base = source_file.replace(".txt", "")
        key = f"{OUTPUT_PREFIX}{base}.jsonl"
        body = s3_client.get_object(Bucket=DATA_BUCKET_NAME, Key=key)["Body"].read().decode("utf-8")
        rows = []
        for line in body.splitlines():
            if not line.strip(): continue
            rows.append(json.loads(line))
        return rows

    # cache 読みすぎ防止
    cache_all_by_src = {}
    for m in top_matches:
        src = m["source_file"]
        if src and src not in cache_all_by_src:
            try:
                cache_all_by_src[src] = _load_all_for_source(src)
            except Exception:
                cache_all_by_src[src] = []

    # pair_id ごとにQ/A束ね
    qa_pairs = []
    seen_pairs = set()
    for m in top_matches:
        src = m["source_file"]
        pid = m.get("pair_id")
        if src and pid is not None and (src, pid) not in seen_pairs:
            seen_pairs.add((src, pid))
            group = [x for x in cache_all_by_src.get(src, []) if x.get("pair_id") == pid]
            q = [x for x in group if x.get("qa_role") == "Q"]
            a = [x for x in group if x.get("qa_role") == "A"]
            qa_pairs.append({"pair_id": pid, "source_file": src, "Q": q, "A": a})

    # ペアごとの要約 → 全体要約（既存UIのまま）
    try:
        gikai_pair_prompt = load_prompt("gikai_pair_summary.txt")
        summary_overall_prompt = load_prompt("gikai_summary_overall.txt")
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

        summary_per_pair = []
        for pair in qa_pairs:
            q_texts = [q["text"] for q in pair["Q"]]
            a_texts = [a["text"] for a in pair["A"]]
            qa_ctx = "\n\n".join(["〖質問〗"+q for q in q_texts] + ["〖答弁〗"+a for a in a_texts])
            if not qa_ctx.strip():
                pair["summary"] = "⚠️ 該当Q/A本文なし"
                continue
            resp = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "system", "content": gikai_pair_prompt},
                          {"role": "user", "content": qa_ctx}],
                temperature=GPT_TEMPERATURE
            )
            s = resp.choices[0].message.content.strip()
            pair["summary"] = s
            summary_per_pair.append(s)

        if summary_per_pair:
            ctx = "\n\n".join([f"{i+1}件目：{s}" for i,s in enumerate(summary_per_pair)])
            resp = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "system", "content": summary_overall_prompt},
                          {"role": "user", "content": ctx}],
                temperature=GPT_TEMPERATURE
            )
            summary_overall = resp.choices[0].message.content.strip()
        else:
            summary_overall = "⚠️ 情報が見つかりませんでした。"

    except Exception as e:
        summary_overall = f"⚠️ 要約処理でエラー：{e}"

    return {"matches": top_matches, "summary": summary_overall, "qa_pairs": qa_pairs}


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
    placeholder="例：学校の給食費無償化についてどのような議論が行われていますか",
    on_change=lambda: st.session_state.update(send_now=True)
)

# --- サジェスト ---
if not st.session_state.get("clarify_active", False):  
    suggestions_master = [
        "公共施設の統廃合について気になります",
        "行政のDX化は進んでいますか",
        "観光・インバウンド対応について教えて"
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
            st.session_state.clarify_active = False  
            with st.spinner(f"⏳ 「{s}」に回答中... 少々お待ちください"):
                results = search_s3vector_and_respond(s, top_k)
                st.session_state.last_answer = results["summary"]
                st.session_state.last_matches = results["matches"]
                st.session_state.qa_pairs = results["qa_pairs"]
                
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
        results = search_s3vector_and_respond(st.session_state.input, top_k)
        st.session_state.last_answer = results["summary"]
        st.session_state.last_matches = results["matches"]
        st.session_state.qa_pairs = results["qa_pairs"]
        
        # ログ記録
        try:
            log_to_gsheet(st.session_state.input, results["summary"])
        except Exception as e:
            st.warning(f"⚠️ ログ記録に失敗しました: {e}")
            
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

        st.markdown(f"---\n\n##### {i}. {summary}")
        
        for q in pair.get("Q", []):
            with st.expander(f"🟢【質問】{q.get('speaker_role')} {q.get('speaker')}（{q.get('source_file', '').replace('.txt', '')}）"):
                st.markdown(q.get("text", ""))

        for a in pair.get("A", []):
            with st.expander(f"🔵【答弁】{a.get('speaker_role')} {a.get('speaker')}（{a.get('source_file', '').replace('.txt', '')}）"):
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
