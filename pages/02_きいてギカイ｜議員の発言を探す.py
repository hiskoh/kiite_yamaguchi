# -*- coding: utf-8 -*-
import streamlit as st
import chardet
from openai import OpenAI
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from datetime import datetime
import random, json, io, re, boto3
import numpy as np
from googleapiclient.http import MediaIoBaseDownload

st.set_page_config(page_title="きいてギカイやまぐち（β）", layout="wide", page_icon="📜")

# ====== ▼ 初期値設定 ========================================================

# 検索パラメータ
TOP_N_RETURN     = 10   # UIに返す上位件数
TOPK_CANDIDATES  = 40   # S3Vectorsから仮取得する候補の数（TOP_N_RETURN より多めに）
SIM_THRESHOLD    = 0.10 # 類似度しきい値（0.0〜1.0, 高いほど類似）

# ChatGPT
GPT_MODEL        = "gpt-4.1-mini"
GPT_TEMPERATURE  = 0.1
EMBED_MODEL      = "text-embedding-3-small"

# AWS
AWS_REGION       = "us-west-2"
OUTPUT_PREFIX    = "council_chunk_jsonl/"  # ← 実データに合わせる
AWS_ACCESS_KEY_S = st.secrets["AWS-KEY"]["AWS_ACCESS_KEY"]
AWS_SECRET_KEY_S = st.secrets["AWS-KEY"]["AWS_SECRET_KEY"]
DATA_BUCKET_NAME = st.secrets["AWS-KEY"]["DATA_BUCKET_NAME"]  # バケット名のみ
S3_INDEX_ARN     = st.secrets["AWS-KEY"]["VECTOR_INDEX_ARN_COUNCIL"]

# ========= ▼ S3 Vectors 検索ユーティリティ ===============================

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
    chunk_id から該当 JSONL を決定し、該当行を返す。
    （speaker/speaker_role/pair_id/qa_role を含む行）
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

# --- 追加：S3 Selectで pair_id 一致行だけ抽出 ---
def _s3select_pair_records(s3_client, jsonl_key: str, pair_id: int):
    """
    JSON Lines の議事録ファイル(jsonl_key)から、指定 pair_id の行のみを抽出。
    Q→A→N の順にソートして返す。
    """
    expr = f"SELECT * FROM S3Object s WHERE s.pair_id = {int(pair_id)}"
    resp = s3_client.select_object_content(
        Bucket=DATA_BUCKET_NAME,
        Key=jsonl_key,
        ExpressionType="SQL",
        Expression=expr,
        InputSerialization={"JSON": {"Type": "LINES"}},
        OutputSerialization={"JSON": {}},
    )
    records = []
    for event in resp["Payload"]:
        if "Records" in event:
            chunk = event["Records"]["Payload"].decode("utf-8")
            for line in chunk.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    records.append(obj)
                except Exception:
                    pass
    order = {"Q": 0, "A": 1, "N": 2}
    records.sort(key=lambda r: order.get(r.get("qa_role"), 9))
    return records

def _query_s3vectors(query_text: str, top_k_: int, score_threshold: float):
    """
    S3 Vectors を検索し、[{score, distance, key, source_file, chunk_id, pair_id, original}] を返す。
    original には JSONL の当該オブジェクト（speaker/qa_role等）が入る。
    """
    # クエリ埋め込み
    oai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    emb = oai.embeddings.create(model=EMBED_MODEL, input=query_text)
    qvec = [float(x) for x in emb.data[0].embedding]

    # Boto3 clients
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

    # 候補を多めに取得 → しきい値でフィルタ
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
        source   = md.get("source_file")   # 表示用には残す
        chunk_id = md.get("chunk_id") or key
        pair_id  = md.get("pair_id")       # ← 追加：メタから pair_id を取得
        original = _fetch_original_chunk_for_search(s3_client, chunk_id)

        out.append({
            "score": score,
            "distance": distance,
            "key": key,
            "source_file": source,
            "chunk_id": chunk_id,
            "pair_id": pair_id,        # ← 追加：ヒットに pair_id を持たせる
            "original": original,
        })

    out.sort(key=lambda x: x["score"], reverse=True)
    return out

# ========= ▼ セッション初期化 =============================================

for key in ["agreed", "query", "send_now", "last_answer", "last_matches", "is_generating",
            "input", "input_value", "suggestions_sampled", "qa_pairs", "clarified", "clarify_active"]:
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

# ========= ▼ 各種ユーティリティ ==========================================

def load_prompt(filename, default_text=""):
    try:
        with open("prompts/" + filename, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return default_text

# 同意画面
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

# OpenAIクライアント
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Google Sheets ログ
def get_gspread_client():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gsheets_service_account"], scope)
    return gspread.authorize(creds)

def log_to_gsheet(question, answer):
    try:
        client_gs = get_gspread_client()
        sheet = client_gs.open_by_key(st.secrets["kiite-gikai"]["GOOGLE_GIKAI_LOG_SHEET_ID"]).worksheet("logs")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([now, question, answer])
    except Exception as e:
        st.warning(f"⚠️ ログ記録に失敗しました: {e}")

# Google Drive から .txt を読む関数（今回の検索では未使用・将来用のまま残置）
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

def on_enter():
    if st.session_state.input.strip():
        st.session_state.send_now = True

def clarify_query(user_query):
    clarify_prompt = load_prompt("gikai_clarify_prompt.txt", default_text="あなたはユーザーの曖昧な質問を明確化します。必要なら例示して書き換え案を出してください。JSONで返してください。")
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
        result = json.loads(content)
        return result
    except Exception as e:
        st.error(f"Clarifyエラー: {e}")
        return {"ambiguous": False, "reason": "", "rewritten_query": ""}

# ========= ▼ 検索〜要約 本体 ===============================================

def search_s3vector_and_respond(query):
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

    # 表示用整形：QAペア抽出に必要な speaker/qa_role/pair_id を original or metadata から取り出す
    top_matches = []
    for h in top_hits:
        o = h.get("original") or {}
        jsonl_base = _base_from_chunk_id(h.get("chunk_id") or "")
        # ← 変更：pair_id は metadata 優先で使用（h['pair_id']）
        pid = h.get("pair_id")
        if pid is None:
            pid = o.get("pair_id")
        top_matches.append({
            "score": float(h.get("score", 0.0)),
            "source_file": h.get("source_file") or o.get("source_file") or "",
            "chunk_id": h.get("chunk_id"),
            "jsonl_base": jsonl_base,
            "text": o.get("text", ""),
            "speaker": o.get("speaker"),
            "speaker_role": o.get("speaker_role"),
            "pair_id": pid,
            "qa_role": o.get("qa_role"),
        })

    # ---- ここから pair_id を使って S3 Select で Q/A を取得（全文ロードを廃止）----
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_S,
        aws_secret_access_key=AWS_SECRET_KEY_S,
        region_name=AWS_REGION,
    )

    # (jsonl_base, pair_id) でユニーク化
    pair_keys = []
    seen = set()
    for m in top_matches:
        jb  = m.get("jsonl_base")
        pid = m.get("pair_id")
        if jb and (pid is not None):
            key_tuple = (jb, int(pid))
            if key_tuple not in seen:
                seen.add(key_tuple)
                pair_keys.append(key_tuple)

    # search_s3vector_and_respond 内：S3 Select実行部分を置換
    qa_pairs = []
    for m in top_matches:
        jb  = m.get("jsonl_base")
        pid = m.get("pair_id")
        src = m.get("source_file")
        if pid is None:
            continue

        jsonl_keys = _guess_jsonl_keys(jb, src, OUTPUT_PREFIX)

        recs = []
        last_err = None
        for jsonl_key in jsonl_keys:
            try:
                recs = _s3select_pair_records(s3_client, jsonl_key, int(pid))
                if recs:
                    break
            except Exception as e:
                last_err = e
                continue
        if not recs:
            if last_err:
                st.warning(f"S3 Select失敗: {jsonl_keys} pair_id={pid} :: {last_err}")
            continue

        Q = [r for r in recs if r.get("qa_role") == "Q"]
        A = [r for r in recs if r.get("qa_role") == "A"]
        src_name = recs[0].get("source_file") if recs else (src or (jb + ".txt"))
        for r in Q + A:
            r.setdefault("source_file", src_name)
        qa_pairs.append({"pair_id": int(pid), "source_file": src_name, "jsonl_base": jb, "Q": Q, "A": A})


    # ペアごとの要約 → 全体要約
    try:
        gikai_pair_prompt = load_prompt("gikai_pair_summary.txt",
                                        "あなたは議会の議事録編集者です。質問と答弁を読み、論点・合意・宿題を箇条書きで短くまとめてください。")
        summary_overall_prompt = load_prompt("gikai_summary_overall.txt",
                                             "複数のQ/A要約を統合し、重複をまとめて全体像を100〜200字でまとめてください。")

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
            s = (resp.choices[0].message.content or "").strip()
            pair["summary"] = s
            summary_per_pair.append(s)

        if summary_per_pair:
            ctx = "\n\n".join([f"{i+1}件目：{s}" for i, s in enumerate(summary_per_pair)])
            resp = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "system", "content": summary_overall_prompt},
                          {"role": "user", "content": ctx}],
                temperature=GPT_TEMPERATURE
            )
            summary_overall = (resp.choices[0].message.content or "").strip()
        else:
            summary_overall = "⚠️ 情報が見つかりませんでした。"

    except Exception as e:
        summary_overall = f"⚠️ 要約処理でエラー：{e}"

    return {"matches": top_matches, "summary": summary_overall, "qa_pairs": qa_pairs}

# ========= ▼ UI =============================================================

st.title("📜 きいてギカイやまぐち（β）")

# 入力欄
st.markdown("---\n\n#### 💬 質問してみよう")
st.text_input(
    label="",
    key="input",
    value=st.session_state.input_value,
    placeholder="例：学校の給食費無償化についてどのような議論が行われていますか",
    on_change=lambda: st.session_state.update(send_now=True)
)

# サジェスト
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
                results = search_s3vector_and_respond(s)
                st.session_state.last_answer = results["summary"]
                st.session_state.last_matches = results["matches"]
                st.session_state.qa_pairs = results["qa_pairs"]
                log_to_gsheet(s, results["summary"])
            st.session_state.is_generating = False
            st.rerun()

# Clarify
if st.session_state.input and not st.session_state.get("clarified", False):
    clarify_result = clarify_query(st.session_state.input)
    if clarify_result["ambiguous"] and clarify_result["rewritten_query"]:
        st.session_state.clarify_active = True
        st.info(f"👇 より正確な検索のため「{clarify_result['reason']}」ことをお勧めします。\n\n**→ {clarify_result['rewritten_query']}**")

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
        st.session_state.clarified = True

# Enter送信
if st.session_state.input and st.session_state.send_now:
    st.session_state.send_now = False
    st.session_state.is_generating = True
    with st.spinner(f"⏳ 「{st.session_state.input}」に回答中... 少々お待ちください"):
        results = search_s3vector_and_respond(st.session_state.input)
        st.session_state.last_answer = results["summary"]
        st.session_state.last_matches = results["matches"]
        st.session_state.qa_pairs = results["qa_pairs"]
        log_to_gsheet(st.session_state.input, results["summary"])
    st.session_state.input_value = ""
    st.session_state.is_generating = False

# 回答表示
if st.session_state.is_generating:
    st.info("⏳ 回答中... 少々お待ちください")
elif st.session_state.last_answer and (st.session_state.qa_pairs or st.session_state.last_matches):
    st.divider()
    st.caption("入力された質問に対して、AIが類似度が高いと判断した発言をもとに回答します。")
    st.markdown("#### 💡議会質問のまとめ")
    st.success(st.session_state.last_answer)

    # Q/Aがあれば従来どおり表示
    if st.session_state.qa_pairs:
        st.markdown("---\n\n#### 📂 各質問の要約と原文")
        for i, pair in enumerate(st.session_state.qa_pairs, start=1):
            summary = (pair.get("summary") or "").strip()
            if not summary:
                continue
            st.markdown(f"---\n\n##### {i}. {summary}")
            for q in pair.get("Q", []):
                with st.expander(f"🟢【質問】{q.get('speaker_role')} {q.get('speaker')}（{q.get('source_file', '').replace('.txt', '')}）"):
                    st.markdown(q.get("text", ""))
            for a in pair.get("A", []):
                with st.expander(f"🔵【答弁】{a.get('speaker_role')} {a.get('speaker')}（{a.get('source_file', '').replace('.txt', '')}）"):
                    st.markdown(a.get("text", ""))

    # QAが組めなかった場合でも、ヒット断片を表示（デバッグにも有効）
    if not st.session_state.qa_pairs and st.session_state.last_matches:
        st.markdown("---\n\n#### 🔎 ヒットした発言（ペア未形成）")
        for m in st.session_state.last_matches:
            score_pct = f"{m.get('score',0.0)*100:.1f}%"
            with st.expander(f"{m.get('speaker_role','')} {m.get('speaker','')}｜{m.get('source_file','')}｜類似度 {score_pct}｜chunk_id={m.get('chunk_id')}"):
                st.write(m.get("text",""))

elif st.session_state.send_now or st.session_state.input.strip() or st.session_state.query:
    st.warning("⚠️ 情報が見つかりませんでした。")

# フッター
st.divider()
st.caption("""
⚠️ 回答は生成AIによるものであり、正確性を保証するものではありません。  
🙌 本プロジェクトは個人により運営されています。ご支援いただける方はぜひこちらから：  
[💛 codocで支援する](https://codoc.jp/sites/p8cEFlTZQA/entries/MMZnODc1dw)
""")


# --- デバッグ用：JSONLキー推定（chunk_id と source_file の両にらみ）
def _guess_jsonl_keys(jsonl_base: str | None, source_file: str | None, output_prefix: str) -> list[str]:
    cands = []
    if jsonl_base:
        cands.append(f"{output_prefix}{jsonl_base}.jsonl")
    if source_file:
        stem = re.sub(r"\.txt$", "", source_file or "")
        cands.append(f"{output_prefix}{stem}.jsonl")

    # 全半角ゆれの軽い正規化
    def _norm(s: str) -> str:
        return s.replace("（", "(").replace("）", ")").replace("　", " ").strip()

    extras = []
    for k in list(cands):
        base = k.replace(output_prefix, "").replace(".jsonl", "")
        nb = _norm(base)
        if nb != base:
            extras.append(f"{output_prefix}{nb}.jsonl")

    # 重複除去
    return list(dict.fromkeys(cands + extras))
# ========== ▼ デバッグパネル =============================================
with st.expander("🧪 デバッグ：ベクター命中 → JSONLキー推定 → S3 Select 確認", expanded=False):
    if st.session_state.last_matches:
        # 先頭5件を表示（score/distance/metadata）
        debug_rows = []
        for h in st.session_state.last_matches[:5]:
            jsonl_base = re.sub(r"_[0-9]{1,3}$", "", (h.get("chunk_id") or ""))
            source_file = h.get("source_file") or ""
            pid = h.get("pair_id") or (h.get("original") or {}).get("pair_id")
            jsonl_keys = _guess_jsonl_keys(jsonl_base, source_file, OUTPUT_PREFIX)
            debug_rows.append({
                "score": round(float(h.get("score", 0.0)), 4),
                "distance": round(float(h.get("distance", 0.0)), 4),
                "source_file": source_file,
                "chunk_id": h.get("chunk_id"),
                "pair_id": pid,
                "jsonl_base": jsonl_base,
                "jsonl_keys_guess": " | ".join(jsonl_keys),
            })
        st.write("▶ 先頭ヒットのメタ確認（最大5件）")
        st.dataframe(debug_rows, use_container_width=True)

        # JSONLの存在確認（HEAD）＆ S3 Select をその場テスト
        s3_client_dbg = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_S,
            aws_secret_access_key=AWS_SECRET_KEY_S,
            region_name=AWS_REGION,
        )

        # テスト対象の1件を選ぶ
        h0 = st.session_state.last_matches[0]
        jb0 = re.sub(r"_[0-9]{1,3}$", "", (h0.get("chunk_id") or ""))
        src0 = h0.get("source_file") or ""
        pid0 = h0.get("pair_id") or (h0.get("original") or {}).get("pair_id")
        jsonl_keys0 = _guess_jsonl_keys(jb0, src0, OUTPUT_PREFIX)

        st.write("▶ 1件テスト対象")
        st.json({
            "source_file": src0,
            "chunk_id": h0.get("chunk_id"),
            "pair_id": pid0,
            "jsonl_base": jb0,
            "jsonl_keys_guess": jsonl_keys0
        })

        # S3上に実在するキーを特定
        existing = []
        for key in jsonl_keys0:
            try:
                s3_client_dbg.head_object(Bucket=DATA_BUCKET_NAME, Key=key)
                existing.append(key)
            except Exception as e:
                pass

        st.write("▶ S3上で存在確認できた JSONL キー", existing if existing else "（なし）")

        # ボタンでS3 Selectを実行（pair_idは負値やNoneはスキップ）
        if st.button("この1件で S3 Select テスト実行"):
            if (pid0 is None) or (int(pid0) < 0):
                st.error(f"pair_id={pid0} は無効（None/負数）。別のヒットを選んでください。")
            else:
                recs = []
                errs = []
                for key in (existing or jsonl_keys0):
                    try:
                        recs = _s3select_pair_records(s3_client_dbg, key, int(pid0))
                        if recs:
                            st.success(f"S3 Select成功: s3://{DATA_BUCKET_NAME}/{key} pair_id={pid0}")
                            # 取得件数やQ/A内訳
                            Q = [r for r in recs if r.get("qa_role") == "Q"]
                            A = [r for r in recs if r.get("qa_role") == "A"]
                            N = [r for r in recs if r.get("qa_role") == "N"]
                            st.write({"Q": len(Q), "A": len(A), "N": len(N), "total": len(recs)})
                            # 先頭のQ/Aを確認用に表示
                            if Q:
                                st.markdown(f"**Q（先頭1件）**: {Q[0].get('speaker_role','')} {Q[0].get('speaker','')}")
                                st.code(Q[0].get("text","")[:500])
                            if A:
                                st.markdown(f"**A（先頭1件）**: {A[0].get('speaker_role','')} {A[0].get('speaker','')}")
                                st.code(A[0].get("text","")[:500])
                            break
                    except Exception as e:
                        errs.append(f"{key} :: {e}")
                        continue
                if not recs:
                    st.error("S3 Selectで該当レコードが見つかりませんでした。")
                    if errs:
                        st.write("試行ログ", errs)
    else:
        st.info("検索ヒットがまだありません。質問を投げた後にご確認ください。")

# ========== ▼ 環境チェック（プローブ：実クエリ1回で生応答を確認） ======================
with st.expander("🧪 環境チェック：埋め込み次元 & S3 Vectors プローブ", expanded=False):
    try:
        # 1) 埋め込み次元を表示
        _client_oai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        _emb = _client_oai.embeddings.create(model=EMBED_MODEL, input="ping")
        _vec = [float(x) for x in _emb.data[0].embedding]
        dim = len(_vec)
        st.write({"embedding_model": EMBED_MODEL, "embedding_dim": dim})

        # 2) プローブ用の疑似ベクトルで indexArn に対して生クエリ
        s3v_dbg = boto3.client(
            "s3vectors",
            aws_access_key_id=AWS_ACCESS_KEY_S,
            aws_secret_access_key=AWS_SECRET_KEY_S,
            region_name=AWS_REGION,
        )

        # 単純な one-hot ベクトルでOK（次元不一致ならここでエラーになる）
        probe = [0.0] * dim
        probe[0] = 1.0

        res = s3v_dbg.query_vectors(
            indexArn=S3_INDEX_ARN,
            queryVector={"float32": probe},
            topK=3,
            returnMetadata=True,
            returnDistance=True,
        )

        vecs = res.get("vectors", []) or []
        sample = vecs[0] if vecs else {}
        # 生応答のキーとサンプル（重すぎない範囲で）
        st.write({
            "probe_vectors_len": len(vecs),
            "probe_first_keys": list(sample.keys()) if sample else [],
            "probe_first_sample": {
                "distance": sample.get("distance"),
                "score": sample.get("score"),
                "key": sample.get("key") or sample.get("id"),
                "metadata_keys": list((sample.get("metadata") or {}).keys()) if sample else [],
            }
        })

    except Exception as e:    # --- ▼ デバッグ出力：クエリ応答の生構造 & サンプル値 -----------------
    try:
        _vecs = res.get("vectors", []) or []
        _first = _vecs[0] if _vecs else {}
        st.write({
            "DEBUG/query_vectors": {
                "topK_used": max(TOPK_CANDIDATES, top_k_),
                "qvec_dim": len(qvec),
                "vectors_len": len(_vecs),
                "first_vector": {
                    "distance": _first.get("distance"),
                    "score": _first.get("score"),
                    "key": _first.get("key") or _first.get("id"),
                    "metadata_keys": list((_first.get("metadata") or {}).keys()) if _first else [],
                }
            }
        })
    except Exception:
        pass

        st.error(f"プローブ失敗: {e}")
        st.info("→ 典型原因: indexArn/リージョン不一致、次元不一致、インデックス未投入、認可エラー")

