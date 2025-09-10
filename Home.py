# Home.py もしくは pages/00_トップページ.py
import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(page_title="きいてポータル｜やまぐち ことばアーカイブ", page_icon="🗣️", layout="wide")

NOTION_URL = "https://fortune-orangutan-6aa.notion.site/1d0311267344808db873ff8af9b67365"
APP_MAYOR_PATH   = "pages/01_きいてミライ｜市長の発言を探す.py"
APP_COUNCIL_PATH = "pages/02_きいてギカイ｜議員の発言を探す.py"
APP_SUMMARY_PATH = "pages/03_頻出発言ダッシュボード｜ことばの傾向を知る.py"

# ---------- CSS（カード & 透明ボタンのオーバーレイ） ----------
st.markdown("""
<style>
.small-muted { color: rgba(0,0,0,0.55); font-size: .9rem; }
.hero { padding: .4rem 0 .2rem 0; }

.card {
  position: relative;               /* ← オーバーレイの基準 */
  padding: 1.1rem;
  border: 1px solid rgba(0,0,0,.12);
  border-radius: 14px;
  background: #fff;
  box-shadow: 0 1px 4px rgba(0,0,0,.04);
  transition: transform .08s ease, box-shadow .12s ease, border-color .12s ease;
  height: 100%;
}
.card:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 18px rgba(0,0,0,.08);
  border-color: rgba(15,103,255,.35);
}
.kicker {
  font-size: .9rem; letter-spacing: .04em; font-weight: 700;
  display:inline-block; padding:.18rem .5rem; border-radius:999px;
  border:1px solid #D5E2FF; background:#EEF3FF; color:#1a57ff;
}

/* カード内に置いた st.button を“透明オーバーレイ”化して、カード全面をクリック可能にする */
.card .stButton > button {
  position: absolute; inset: 0;     /* カード全面を覆う */
  opacity: 0;                       /* 完全透明（見た目は消える）*/
  cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

st.title("きいてポータル｜やまぐち ことばアーカイブ")
st.markdown('<div class="hero small-muted">市長や議員の発言を検索・分析できるサイトです。政策やまちづくりに関する議論を、もっと身近に。</div>', unsafe_allow_html=True)
st.divider()

# ---------- ヘルパー（カード=クリックで遷移） ----------
def card_navigate(page_py: str, kicker: str, title: str, desc: str, key: str):
    # カード本体（見た目）
    st.markdown(f"""
    <div class="card">
      <div class="kicker">{kicker}</div>
      <div style="font-size:1.15rem; font-weight:700; margin:.2rem 0 0 0;">{title}</div>
      <p style="color:rgba(0,0,0,.65); line-height:1.5; font-size:.95rem; margin:0;">
        {desc}
      </p>
    </div>
    """, unsafe_allow_html=True)

    # 透明ボタン（カード全面を覆う）
    if hasattr(st, "switch_page"):
        if st.button(" ", key=key):           # ラベルは空でOK（CSSで透明）
            st.switch_page(page_py)
    else:
        # 古い環境用フォールバック（ボタン表示でOKならこちら）
        st.page_link(page_py, label="➡️ ページを開く")

# ---------- 3カード ----------
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    card_navigate(
        page_py=APP_MAYOR_PATH,
        kicker="👔 市長の発言を探す",
        title="聞いてミライ",
        desc="施政方針や記者会見をRAGで検索。タグ・年度で絞り込み、要点要約で素早く把握できます。",
        key="go_mayor"
    )

with col2:
    card_navigate(
        page_py=APP_COUNCIL_PATH,
        kicker="🏛 議員の発言を探す",
        title="聞いてギカイ",
        desc="会派・議員名・定例会で検索。質問と答弁のペア表示で、議論の流れが一目で分かります。",
        key="go_council"
    )

with col3:
    card_navigate(
        page_py=APP_SUMMARY_PATH,
        kicker="📊 ことばの傾向を知る",
        title="頻出発言ダッシュボード",
        desc="頻出ワード、共起ネットワーク、テーマの時系列推移、質問スタイル分析などを可視化。",
        key="go_summary"
    )

st.divider()

# ---------- 概要 & Notion ----------
st.subheader("このサイトについて")
st.write(
    "「きいてポータル」は、**市長や議員の発言を検索・分析できるアーカイブ**です。"
    " 政治やまちづくりの議論は、多くの人にとって「遠い存在」になりがちです。"
    " このサイトを通じて少しでも市政に関する議論を身近に、もっと分かりやすく届けることを目指しています。\n\n"
    "- **市長の発言**（施政方針・記者会見など）\n"
    "- **議員の発言**（定例会での質問や答弁）\n"
    "- **発言まとめ**（頻出ワードや傾向分析）\n\n"
    "を収録し、検索や可視化を通じて「市政の見える化」に取り組んでいます。"
)
st.write(f'🔗[本プロジェクトの詳細はこちら]({NOTION_URL})')

# ---------- フッター（JST） ----------
st.markdown("---")
c1, c2, c3, c4 = st.columns([1,1,1,2])
with c1: st.write("© 2025 きいてポータル")
with c2: st.write("Made with Streamlit")
with c4:
    jst_now = datetime.now(ZoneInfo("Asia/Tokyo"))
    st.markdown(f'<span class="small-muted">最終更新: {jst_now.strftime("%Y-%m-%d %H:%M")} JST</span>', unsafe_allow_html=True)
