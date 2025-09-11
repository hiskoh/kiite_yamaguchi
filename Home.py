import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

# -----------------------------
# 基本設定
# -----------------------------
st.set_page_config(
    page_title="きいてポータル｜やまぐち市政を聞いてみよう",
    page_icon="🗣️",
    layout="wide"
)

NOTION_URL = "https://fortune-orangutan-6aa.notion.site/1d0311267344808db873ff8af9b67365"
APP_MAYOR_PATH   = "pages/01_きいてミライ｜市長の発言を探す.py"
APP_COUNCIL_PATH = "pages/02_きいてギカイ｜議員の発言を探す.py"
APP_SUMMARY_PATH = "pages/03_頻出発言ダッシュボード.py"

# -----------------------------
# スタイル（カード＋リンク風ボタン）
# -----------------------------
st.markdown("""
<style>
.small-muted { color: rgba(0,0,0,0.55); font-size: .9rem; margin: 0 0 .8rem 0; }
.hero { padding: .4rem 0 .2rem 0; }

/* container(border=True) をカードっぽく */
div[data-testid="stContainer"] > div:has(> .card-inner) {
  padding: 1.1rem !important;
  border: 1px solid rgba(0,0,0,.12) !important;
  border-radius: 14px !important;
  background: #fff !important;
  box-shadow: 0 1px 4px rgba(0,0,0,.04) !important;
  transition: transform .08s ease, box-shadow .12s ease, border-color .12s ease;
}
div[data-testid="stContainer"] > div:has(> .card-inner):hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 18px rgba(0,0,0,.08);
  border-color: rgba(15,103,255,.35);
}

/* バッジ */
.kicker {
  font-size: .9rem; letter-spacing: .04em; font-weight: 700;
  display:inline-block; padding:.18rem .5rem; border-radius:999px;
  border:1px solid #D5E2FF; background:#EEF3FF; color:#1a57ff;
}

/* ▼ リンク前の余白用ラッパ */
.link-wrap { 
  margin-top: .12rem !important;    /* 半行程度の余白 */
}

/* ▼ st.page_link を“グレーのリンク風”に強制上書き */
.link-wrap a[data-testid="stPageLink"]{
  display: block !important;
  width: 100% !important;
  text-align: left !important;
  padding: .4rem .2rem !important;
  background: transparent !important;
  border: none !important;
  color: rgba(0,0,0,.55) !important;      /* グレー文字 */
  font-weight: 500 !important;
  text-decoration: none !important;
}
.link-wrap a[data-testid="stPageLink"]:hover{
  color: rgba(15,103,255,.85) !important;  /* hoverで青く */
  text-decoration: underline !important;   /* 下線でリンク感 */
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 安全なページ遷移ヘルパー
# -----------------------------
def page_link_safe_mp(page_py: str, label: str, icon: str = "➡️"):
    """
    Multipageで安全にページ遷移するためのヘルパー。
    1) ページ名（サイドバー表示名）で st.page_link
    2) だめならファイルパスで st.page_link
    3) だめなら switch_page
    4) 最後に link_button
    """
    page_name = Path(page_py).stem  # 例: "01_きいてミライ｜市長の発言を探す"
    try:
        st.page_link(page_name, label=label, icon=icon)
        return
    except Exception:
        pass
    try:
        st.page_link(page_py, label=label, icon=icon)
        return
    except Exception:
        pass
    if hasattr(st, "switch_page"):
        if st.button(f"{icon} {label}", use_container_width=True):
            st.switch_page(page_py)
    else:
        st.link_button(f"{icon} {label}", url="#")

# -----------------------------
# カード描画（カード内に“リンク風”ボタン）
# -----------------------------
def card_with_link(page_py: str, kicker: str, title: str, desc: str, link_label: str):
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="card-inner">
              <div class="kicker">{kicker}</div>
              <div style="font-size:1.15rem; font-weight:700; margin:.2rem 2rem 0 0;">
                {title}
              </div>
              <p style="color:rgba(0,0,0,.65); line-height:1.5; font-size:.95rem; margin:.4rem 0 0 0;">
                {desc}
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # ▼ ここで page_link をラップして余白＋スタイルを確実に適用
        st.markdown('<div class="link-wrap">', unsafe_allow_html=True)
        page_link_safe_mp(page_py, link_label)
        st.markdown('</div>', unsafe_allow_html=True)    
        
# -----------------------------
# ヘッダー
# -----------------------------
st.title("💭 きいてポータル")
st.markdown('<div class="hero small-muted">市長や議員の発言を検索・分析できるサイトです。政策やまちづくりに関する議論を、もっと身近に。 </div>', unsafe_allow_html=True)

# -----------------------------
# 3カード
# -----------------------------
c1, c2, c3 = st.columns(3, gap="large")

with c1:
    card_with_link(
        page_py=APP_MAYOR_PATH,
        kicker="👔 市長の発言を探す",
        title="きいてミライ",
        desc="施政方針や記者会見をRAGで検索。要点要約で素早く把握できます。",
        link_label="市長の発言を見る",
    )

with c2:
    card_with_link(
        page_py=APP_COUNCIL_PATH,
        kicker="🏛 議員の発言を探す",
        title="きいてギカイ",
        desc="会派・議員名・定例会で検索。質問と答弁のペア表示で、議論の流れが一目で分かります。",
        link_label="議員の発言を見る",
    )

with c3:
    card_with_link(
        page_py=APP_SUMMARY_PATH,
        kicker="📊 ことばの傾向を知る",
        title="頻出発言ダッシュボード",
        desc="頻出ワード、共起ネットワーク、テーマの時系列推移、質問スタイル分析などを可視化。",
        link_label="発言をまとめて見る",
    )

st.divider()

# -----------------------------
# 概要 & Notionリンク
# -----------------------------
st.subheader("このサイトについて")
st.write(
    "「きいてポータル」は、**市長や議員の発言を検索・分析できるアーカイブ**です。\n"
    " 政治やまちづくりの議論は、多くの人にとって「遠い存在」になりがちです。\n"
    " このサイトでは生成AI等のIT技術を活用し、少しでも市政を身近に、もっと分かりやすく届けることを目指しています。\n\n"
)
st.write(f'🔗[本プロジェクトの詳細を知りたい方はこちらもご覧ください！]({NOTION_URL})')

# -----------------------------
# フッター（JST）
# -----------------------------
st.markdown("---")
col_a, col_b, col_c, col_d = st.columns([1,1,1,2])
with col_a: st.write("© 2025 きいてポータル")
with col_b: st.write("Made with Streamlit")
with col_d:
    jst_now = datetime.now(ZoneInfo("Asia/Tokyo"))
    st.markdown(f'<span class="small-muted">最終更新: {jst_now.strftime("%Y-%m-%d %H:%M")} JST</span>', unsafe_allow_html=True)
