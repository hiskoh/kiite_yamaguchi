# Home.py もしくは pages/00_トップページ.py
import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo

# -----------------------------
# 基本設定
# -----------------------------
st.set_page_config(
    page_title="きいてポータル｜やまぐち ことばアーカイブ",
    page_icon="🗣️",
    layout="wide"
)

NOTION_URL = "https://fortune-orangutan-6aa.notion.site/1d0311267344808db873ff8af9b67365"
APP_MAYOR_PATH = "pages/01_きいてミライ｜市長の発言を探す.py"               # 市長ページのパス
APP_COUNCIL_PATH = "pages/02_きいてギカイ｜議員の発言を探す.py"             # 議員ページのパス
APP_SUMMARY_PATH = "pages/03_頻出発言ダッシュボード｜ことばの傾向を知る.py"  # 発言まとめページのパス

# -----------------------------
# スタイル（カード枠＋軽いホバー）
# -----------------------------
st.markdown("""
<style>
.small-muted { color: rgba(0,0,0,0.55); font-size: 0.9rem; }
.hero { padding: 0.4rem 0 0.2rem 0; }
a[href^="https://"] { text-decoration: none; }

/* カード */
.card {
  padding: 1.1rem 1.1rem 1.0rem 1.1rem;
  border: 1px solid rgba(0,0,0,0.12);
  border-radius: 14px;
  background: #fff;
  box-shadow: 0 1px 4px rgba(0,0,0,0.04);
  transition: transform .08s ease, box-shadow .12s ease, border-color .12s ease;
  height: 100%;
  display: flex; flex-direction: column; gap: .6rem;
}
.card:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 18px rgba(0,0,0,0.08);
  border-color: rgba(15,103,255,0.35);
}
.card h3 { margin: .2rem 0 .2rem 0; line-height: 1.22; }

/* セクションラベル */
.kicker {
  font-size: 0.9rem;
  letter-spacing: .04em;
  font-weight: 700;
  display:inline-block;
  padding:.18rem .5rem;
  border-radius:999px;
  border:1px solid #D5E2FF;
  background:#EEF3FF;
  color:#1a57ff;
}

/* ボタン幅を揃える（page_link の見た目差吸収） */
.stButton > button, .stLinkButton > button { width: 100%; border-radius: 12px; padding: .6rem 1rem; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
st.title("きいてポータル｜やまぐち ことばアーカイブ")
st.markdown('<div class="hero small-muted">市長や議員の発言を検索・分析できるサイトです。政策やまちづくりに関する議論を、もっと身近に。</div>', unsafe_allow_html=True)
st.divider()

# -----------------------------
# 3カード（市長／議員／まとめ）
# -----------------------------
col1, col2, col3 = st.columns(3, gap="large")

def page_link_safe(target: str, label: str, icon: str = "➡️"):
    """st.page_link が無い環境へのフォールバック"""
    try:
        st.page_link(target, label=f"{label} {icon}")
    except Exception:
        # Multipage未設定環境では見た目だけのリンクボタン（遷移しません）
        st.link_button(f"{label} {icon}", url="#")
        st.caption("※ このアプリのマルチページ構成でご利用ください。")
        
def card_link(page: str, kicker: str, title: str, subtitle: str, desc: str):
    # カードの上半分（kicker, title, subtitle, 説明）
    st.markdown(f"""
    <div class="card">
        <div class="kicker">{kicker}</div>
        <div style="font-size:1.2rem; font-weight:700; margin:0.2rem 0 0 0;">
            {title}
        </div>
        <div style="font-size:1rem; font-weight:600; margin-bottom:0.6rem;">
            {subtitle}
        </div>
        <p style="color:rgba(0,0,0,0.65); line-height:1.5; font-size:0.95rem; margin-bottom:0.8rem;">
            {desc}
        </p>
    """, unsafe_allow_html=True)

    # リンク部分（カードの中に見せたいので margin を調整）
    page_link_safe(page, f"{subtitle} を見る")

    # カード閉じタグ
    st.markdown("</div>", unsafe_allow_html=True)

with col1:
    card_link(
        APP_MAYOR_PATH,
        "👔 行政",
        "聞いてミライ",
        "市長の発言を探す",
        "施政方針や記者会見をRAGで検索。タグ・年度で絞り込み、要点要約で素早く把握できます。"
    )

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<span class="kicker">🏛 議会</span>', unsafe_allow_html=True)
    st.markdown("### 聞いてギカイ｜議員の発言を探す")
    st.write("会派・議員名・定例会で検索。質問と答弁のペア表示で、議論の流れが一目で分かります。")
    page_link_safe(APP_COUNCIL_PATH, "議員の発言を見る")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<span class="kicker">📊 INSIGHTS</span>', unsafe_allow_html=True)
    st.markdown("### 頻出発言ダッシュボード｜ことばの傾向を知る")
    st.write("頻出ワード、共起ネットワーク、テーマの時系列推移、質問スタイル分析などを可視化。")
    page_link_safe(APP_SUMMARY_PATH, "発言をまとめて見る")
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# -----------------------------
# 概要 & Notionリンク
# -----------------------------
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

# -----------------------------
# フッター（JST）
# -----------------------------
st.markdown("---")
cols = st.columns([1,1,1,2])
with cols[0]:
    st.write("© 2025 きいてポータル")
with cols[1]:
    st.write("Made with Streamlit")
with cols[3]:
    jst_now = datetime.now(ZoneInfo("Asia/Tokyo"))
    st.markdown(
        f'<span class="small-muted">最終更新: {jst_now.strftime("%Y-%m-%d %H:%M")} JST</span>',
        unsafe_allow_html=True
    )
