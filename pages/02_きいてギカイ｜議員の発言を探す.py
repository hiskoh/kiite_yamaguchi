# -*- coding: utf-8 -*-
# シンプル版：GoogleDrive+FAISS を S3 Vectors + .jsonl に移植
# - インデックスのメタデータは {source_id, chank_id(or chunk_id), pair_id} を想定
# - S3上の議事録は .jsonl（圧縮なし）のみ
# - 以前と同じUI/出力構造（qa_pairs優先表示、なければヒット断片）

import streamlit as st
import chardet
from openai import OpenAI
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import random, json, re, boto3
import numpy as np

# ========================= ページ設定 =========================
st.set_page_config(page_title="きいてギカイやまぐち（β）", layout="wide", page_icon="📜")

# ========================= 初期値設定 =========================
# ChatGPT
GPT_MODEL        = "gpt-4.1-mini"
GPT_TEMPERATURE  = 0.1
EMBED_MODEL      = "text-embedding-3-small"   # <- 必要に応じて変更

# 検索パラメータ
TOPK_CANDIDATES  = 40    # S3Vectorsから仮取得する候補の数（TOP_K より多めに）
SIM_THRESHOLD    = 0.00  # まずは0.0で挙動確認→落ち着いたら 0.05〜0.15 へ
# 取得するチャンク数（≒類似度の高い議会答弁を取得する際、何件まで取得するかを制御）
TOP_K = 10

# AWS
AWS_REGION       = "us-west-2"
DATA_BUCKET_NAME = st.secrets["AWS-KEY"]["DATA_BUCKET_NAME"]
OUTPUT_PREFIX    = "council_chunk_jsonl/"   # S3上のjsonl保存プレフィックス（末尾/）
AWS_ACCESS_KEY_S = st.secrets["AWS-KEY"]["AWS_ACCESS_KEY"]
AWS_SECRET_KEY_S = st.secrets["AWS-KEY"]["AWS_SECRET_KEY"]
S3_INDEX_ARN     = st.secrets["AWS-KEY"]["VECTOR_INDEX_ARN_COUNCIL"]

# ========================= セッション初期化 =========================
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

# ========================= ユーティリティ =========================
def load_prompt(filename, default_text=""):
    try:
        with open("prompts/" + filename, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return default_text

def _to_similarity(distance: float) -> float:
    """S3 Vectorsのdistance(=cosine距離) -> 類似度(1 - distance)"""
    try:
        d = float(distance)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, 1.0 - d))

def _to_int_or_none(x):
    if x is None:
        return None
    try:
        return int(float(str(x).strip()))
    except Exception:
        return None

def _base_from_chunk_id(chunk_id: str) -> str:
    # "somefile_001" → "somefile"
    return re.sub(r"_[0-9]{1,3}$", "", chunk_id or "")

def _meta_get(md: dict, primary: str, alts: list[str]):
    """メタデータから primary または代替キーを順に取得"""
    if md is None:
        return None
    if primary in md and md.get(primary) not in (None, ""):
        return md.get(primary)
    for k in alts:
        v = md.get(k)
        if v not in (None, ""):
            return v
    return None


    def _norm(s: str) -> str:
        return s.replace("（", "(").replace("）", ")").replace("　", " ").strip()

    extras = []
    for k in list(cands):
        base = k.replace(OUTPUT_PREFIX, "").replace(".jsonl", "")
        nb = _norm(base)
        if nb != base:
            extras.append(f"{OUTPUT_PREFIX}{nb}.jsonl")

    # 重複除去
    return list(dict.fromkeys(cands + extras))

# ========================= クライアント =========================
client_oai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def _boto_s3():
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_S,
        aws_secret_access_key=AWS_SECRET_KEY_S,
        region_name=AWS_REGION,
    )

def _boto_s3vectors():
    return boto3.client(
        "s3vectors",
        aws_access_key_id=AWS_ACCESS_KEY_S,
        aws_secret_access_key=AWS_SECRET_KEY_S,
        region_name=AWS_REGION,
    )

# ========================= JSONLアクセス（.jsonlのみ） =========================
def _fetch_original_chunk_for_search(s3_client, chunk_id: str) -> dict | None:
    """
    chunk_id から該当 JSONL を決定し、該当行を返す（speaker/role/pair_id/qa_role/text等）。
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
        # chank_id（typo）も拾う
        if obj.get("chunk_id") == chunk_id or obj.get("chank_id") == chunk_id:
            return obj
    return None

def _s3select_pair_records(s3_client, jsonl_key: str, pair_id: int):
    """
    JSON Lines の議事録ファイル(jsonl_key)から、指定 pair_id の行のみを抽出。
    Q→A→N の順でソートして返す（.jsonlのみ）
    """
    expr = f"SELECT * FROM S3Object s WHERE s.pair_id = {int(pair_id)}"
    resp = s3_client.select_object_content(
        Bucket=DATA_BUCKET_NAME,
        Key=jsonl_key,
        ExpressionType="SQL",
        Expression=expr,
        InputSerialization={"JSON": {"Type": "LINES"}},  # .jsonlのみ
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

# ========================= ベクター検索（S3 Vectors） =========================
def _query_s3vectors(query_text: str, top_k: int, score_threshold: float):
    emb = client_oai.embeddings.create(model=EMBED_MODEL, input=query_text)
    qvec = [float(x) for x in emb.data[0].embedding]

    s3_client = _boto_s3()
    s3v_client = _boto_s3vectors()

    res = s3v_client.query_vectors(
        indexArn=S3_INDEX_ARN,
        queryVector={"float32": qvec},
        topK=max(TOPK_CANDIDATES, top_k),
        returnMetadata=True,
        returnDistance=True,
    )
    matches = res.get("vectors", []) or []

    # ここから：デバッグ可視化（フィルタ前）
    st.write({
        "DEBUG_raw_vectors_len": len(matches),
        "DEBUG_raw_top5_dist": [float(m.get("distance", 0.0)) for m in matches[:5]],
        "DEBUG_raw_top5_score(1-d)": [1.0 - float(m.get("distance", 0.0)) for m in matches[:5]],
        "DEBUG_embed_model": EMBED_MODEL,
    })

    out = []
    dropped_low_score = 0

    for m in matches:
        key      = m.get("key") or m.get("id")
        distance = float(m.get("distance", 0.0))
        score    = 1.0 - distance
        if score < score_threshold:
            dropped_low_score += 1
            continue

        md       = m.get("metadata") or {}
        source_id = md.get("source_id") or md.get("source_file") or md.get("source")
        chunk_id  = md.get("chunk_id") or md.get("chank_id") or key
        pair_id   = md.get("pair_id") or md.get("pairId")
        pair_id   = int(float(pair_id)) if pair_id is not None else None

        original = _fetch_original_chunk_for_search(s3_client, chunk_id)

        out.append({
            "score": score,
            "distance": distance,
            "key": key,
            "source_id": source_id,
            "chunk_id": chunk_id,
            "pair_id": pair_id,
            "original": original,
        })

    out.sort(key=lambda x: x["score"], reverse=True)

    # ここまでの集計理由を出力
    st.write({
        "DEBUG_after_filter_len": len(out),
        "DEBUG_dropped_low_score": dropped_low_score,
        "DEBUG_threshold": score_threshold,
        "DEBUG_top5_after": [
            {
                "score": round(h["score"], 4),
                "distance": round(h["distance"], 4),
                "source_id": h["source_id"],
                "chunk_id": h["chunk_id"],
                "pair_id": h["pair_id"],
            } for h in out[:5]
        ]
    })

    # フェイルセーフ：全部落ちたら、しきい値を無視して上位だけ返す（低確度だがUIで見える化）
    if not out and matches:
        fallback = []
        for m in matches[:min(top_k, len(matches))]:
            md  = m.get("metadata") or {}
            fallback.append({
                "score": 1.0 - float(m.get("distance", 0.0)),
                "distance": float(m.get("distance", 0.0)),
                "key": m.get("key") or m.get("id"),
                "source_id": md.get("source_id") or md.get("source_file") or md.get("source"),
                "chunk_id": md.get("chunk_id") or md.get("chank_id") or (m.get("key") or m.get("id")),
                "pair_id": int(float(md.get("pair_id"))) if md.get("pair_id") is not None else None,
                "original": None,
            })
        st.warning("全件がしきい値で除外されたため、低確度フェイルセーフで上位を暫定表示します。")
        return fallback
    return out


# ========================= Google Sheets ログ =========================
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

# ========================= Clarify（任意） =========================
def clarify_query(user_query):
    clarify_prompt = load_prompt(
        "gikai_clarify_prompt.txt",
        default_text="あなたはユーザーの曖昧な質問を明確化します。必要なら例示して書き換え案を出してください。JSONで返してください。"
    )
    messages = [
        {"role": "system", "content": clarify_prompt},
        {"role": "user", "content": f"【質問】{user_query}"}
    ]
    try:
        response = client_oai.chat.completions.create(
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

# ========================= 検索〜要約 本体 =========================
def search_s3vector_and_respond(query):
    # --- ベクター検索 ---
    try:
        hits = _query_s3vectors(query_text=query, top_k=TOP_K, score_threshold=SIM_THRESHOLD)
    except Exception as e:
        return {"matches": [], "summary": f" 検索エラーが発生しました: {e}", "qa_pairs": []}

    if not hits:
        return {"matches": [], "summary": " 関連する議事録の抜粋は見つかりませんでした。", "qa_pairs": []}

    # 表示用整形：QAペア抽出に必要な speaker/qa_role/pair_id を original or metadata から取り出す
    top_matches = []
    for h in hits[:TOP_K]:
        o = h.get("original") or {}
        jsonl_base = _base_from_chunk_id(h.get("chunk_id") or "")
        pid = _to_int_or_none(h.get("pair_id") or o.get("pair_id"))
        top_matches.append({
            "score": float(h.get("score", 0.0)),
            "source_id": h.get("source_id") or o.get("source_id") or o.get("source_file") or "",
            "chunk_id": h.get("chunk_id"),
            "jsonl_base": jsonl_base,
            "text": o.get("text", ""),
            "speaker": o.get("speaker"),
            "speaker_role": o.get("speaker_role"),
            "pair_id": pid,
            "qa_role": o.get("qa_role"),
        })

    # ---- pair_id を使って S3 Select で Q/A を取得 ----
    s3_client = _boto_s3()
    qa_pairs = []

    for m in top_matches:
        jb  = m.get("jsonl_base")
        pid = m.get("pair_id")
        src = m.get("source_id")
        if pid is None:
            continue

        # jsonl_base を直接使ってキーを作る
        jsonl_key = f"{OUTPUT_PREFIX}{jsonl_base}.jsonl"

        recs = []
        try:
            s3_client.head_object(Bucket=DATA_BUCKET_NAME, Key=jsonl_key)
            recs = _s3select_pair_records(s3_client, jsonl_key, int(pid))
        except Exception as e:
            st.warning(f"S3 Select失敗: {jsonl_key} pair_id={pid} :: {e}")


        if not recs:
            continue

        Q = [r for r in recs if r.get("qa_role") == "Q"]
        A = [r for r in recs if r.get("qa_role") == "A"]
        src_name = recs[0].get("source_id") or recs[0].get("source_file") or (src or (jb + ".txt"))
        for r in Q + A:
            r.setdefault("source_file", src_name)
        qa_pairs.append({"pair_id": int(pid), "source_file": src_name, "jsonl_base": jb, "Q": Q, "A": A})

    # --- ペアごとの要約 → 全体要約 ---
    try:
        gikai_pair_prompt = load_prompt(
            "gikai_pair_summary.txt",
            "あなたは議会の議事録編集者です。質問と答弁を読み、論点・合意・宿題を箇条書きで短くまとめてください。"
        )
        summary_overall_prompt = load_prompt(
            "gikai_summary_overall.txt",
            "複数のQ/A要約を統合し、重複をまとめて全体像を100〜200字でまとめてください。"
        )

        summary_per_pair = []
        for pair in qa_pairs:
            q_texts = [q.get("text", "") for q in pair["Q"]]
            a_texts = [a.get("text", "") for a in pair["A"]]
            qa_ctx = "\n\n".join(["〖質問〗"+q for q in q_texts if q] + ["〖答弁〗"+a for a in a_texts if a])
            if not qa_ctx.strip():
                pair["summary"] = "⚠️ 該当Q/A本文なし"
                continue
            resp = client_oai.chat.completions.create(
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
            resp = client_oai.chat.completions.create(
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

# ========================= UI =========================
st.title("📜 きいてギカイやまぐち（β）")

# 同意画面
if not st.session_state.agreed:
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
st.markdown("#### 💡議会質問のまとめ")
if st.session_state.is_generating:
    st.info("⏳ 回答中... 少々お待ちください")
elif st.session_state.last_answer and (st.session_state.qa_pairs or st.session_state.last_matches):
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

    # QAが組めなかった場合でも、ヒット断片を表示
    if not st.session_state.qa_pairs and st.session_state.last_matches:
        st.markdown("---\n\n#### 🔎 ヒットした発言（ペア未形成）")
        for m in st.session_state.last_matches:
            score_pct = f"{m.get('score',0.0)*100:.1f}%"
            with st.expander(f"{m.get('speaker_role','')} {m.get('speaker','')}｜{m.get('source_id','')}｜類似度 {score_pct}｜chunk_id={m.get('chunk_id')}"):
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


# ==================== 単語テスト：S3 Vectors に直問い合わせ ====================
with st.expander("🧪 S3 Vectors 単語テスト（複数チャンク返るかチェック）", expanded=False):
    test_kw = st.text_input("テストする単語（例：給食費）", value="給食費", key="sv_test_kw")
    topk_for_test = st.number_input("topK", min_value=1, max_value=200, value=30, step=1, key="sv_test_topk")
    if st.button("🔍 S3 Vectors にテスト実行", key="sv_test_btn"):
        try:
            _emb = client_oai.embeddings.create(model=EMBED_MODEL, input=test_kw)
            qvec = [float(x) for x in _emb.data[0].embedding]

            s3v = _boto_s3vectors()
            res = s3v.query_vectors(
                indexArn=S3_INDEX_ARN,
                queryVector={"float32": qvec},
                topK=int(topk_for_test),
                returnMetadata=True,
                returnDistance=True,
            )
            matches = res.get("vectors", []) or []

            # 正規化して保存（S3 Selectテストがこの結果を使えるようにする）
            rows = []
            uniq_chunk_ids = set()
            for m in matches:
                md = m.get("metadata") or {}
                chunk_id = md.get("chunk_id") or md.get("chank_id") or (m.get("key") or m.get("id"))
                pair_id  = md.get("pair_id") or md.get("pairId")
                try:
                    pair_id = int(float(pair_id)) if pair_id is not None else None
                except Exception:
                    pair_id = None
                item = {
                    "distance": float(m.get("distance", 0.0)),
                    "score(1-d)": 1.0 - float(m.get("distance", 0.0)),
                    "key": m.get("key") or m.get("id"),
                    "source_id": md.get("source_id") or md.get("source_file") or md.get("source"),
                    "chunk_id": chunk_id,
                    "pair_id": pair_id,
                }
                rows.append(item)
                uniq_chunk_ids.add(chunk_id)

            # セッションに保存（S3 Selectテストが参照）
            st.session_state["sv_test_matches"] = rows
            # UI側の既存ロジックにも乗るように（任意）
            st.session_state["last_matches"] = rows

            st.write({
                "返却ベクトル数": len(matches),
                "ユニークchunk_id数": len(uniq_chunk_ids),
                "埋め込みモデル": EMBED_MODEL,
            })

            if len(uniq_chunk_ids) >= 2:
                st.success("✅ 複数チャンクが返っています。")
            elif len(matches) > 0:
                st.warning("⚠️ 返却はあるが、chunk_id が単一です。投入データ数/メタを確認してください。")
            else:
                st.error("❌ ヒット0件です。インデックス投入/モデル一致/ARN/リージョンをご確認ください。")

            st.caption("上位の結果（scoreは 1 - distance）")
            import pandas as pd
            df = pd.DataFrame(rows)
            st.dataframe(df.head(50), use_container_width=True)

        except Exception as e:
            st.error(f"テスト実行エラー: {e}")
            st.info("よくある原因: EMBED_MODEL不一致 / S3_INDEX_ARNやリージョンの不整合 / インデックス未投入")


# ==================== S3 Select テスト（先頭ヒットで試す） ====================
def _s3select_pair_records_lenient(s3_client, jsonl_key: str, pair_id: int):
    """pair_id が数値か文字列かに関わらずヒットするように OR 条件で検索"""
    pid_int = int(pair_id)
    pid_str = str(pid_int)
    expr = (
        "SELECT * FROM S3Object s "
        f"WHERE s.pair_id = {pid_int} OR CAST(s.pair_id AS VARCHAR) = '{pid_str}'"
    )
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

with st.expander("🧪 S3 Select テスト（先頭ヒットで試す）", expanded=False):
    # 単語テストの結果 -> 既存last_matches の順で参照
    cand = st.session_state.get("sv_test_matches") or st.session_state.get("last_matches") or []
    if cand:
        s3_client_dbg = _boto_s3()
        h0 = cand[0]
        jb0 = _base_from_chunk_id(h0.get("chunk_id") or "")
        src0 = h0.get("source_id")
        pid0 = h0.get("pair_id")
        jsonl_keys0 = [f"{OUTPUT_PREFIX}{jsonl_base}.jsonl"]
        st.write("▶ 先頭ヒット:", {"pair_id": pid0, "chunk_id": h0.get("chunk_id"), "source_id": src0})
        st.write("▶ 推定JSONLキー:", jsonl_keys0)

        # S3存在チェック
        exist_logs = []
        for key in jsonl_keys0:
            try:
                s3_client_dbg.head_object(Bucket=DATA_BUCKET_NAME, Key=key)
                exist_logs.append((key, True))
            except Exception:
                exist_logs.append((key, False))
        st.write("▶ S3存在確認:", exist_logs)

        if st.button("S3 Select 実行", key="btn_s3sel"):
            if pid0 is None:
                st.error("pair_id が None のため実行不可")
            else:
                found = False
                for key in jsonl_keys0:
                    try:
                        recs = _s3select_pair_records_lenient(s3_client_dbg, key, int(pid0))
                        if recs:
                            st.success(f"S3 Select成功: {len(recs)}件")
                            st.json(recs[:3])
                            found = True
                            break
                    except Exception as e:
                        st.error(f"{key} :: {e}")
                if not found:
                    st.error("S3 Selectで該当レコードが見つかりませんでした。")
    else:
        st.info("まだ検索していません。まず単語テストを実行してください。")


