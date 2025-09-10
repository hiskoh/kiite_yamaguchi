# Home.py もしくは pages/00_トップページ.py
import streamlit as st
from datetime import datetime

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
# CSS（行間・見出し・カード調整）
# -----------------------------
st.markdown("""
<style>
/* h1サイズと余白をやや小さく */
h1 { font-size: 2.4rem !important; margin-bottom: .2rem !important; }
.block-container { padding-top: 2rem; }

/* 説明文のトーン */
.small-muted { color: rgba(0,0,0,0.60); font-size: 0.95rem; }

/* カード */
.card {
  padding: 1.0rem 1.1rem 1.1rem 1.1rem;
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 16px;
  box-shadow: 0 1px 6px rgba(0,0,0,0.06);
  background: #fff;
  height: 100%;
}
.kicker {
  font-size: 0.92rem; letter-spacing: .04em; color: #0F67FF; font-weight: 700;
  margin-top: .2rem; margin-bottom: .2rem;
}
.card h3 { margin: .2rem 0 .4rem 0; line-height: 1.25; }

/* ボタンを幅いっぱいにしてズレ防止 */
.stButton > button { width: 100%; border-radius: 12px; padding: .6rem 1rem; }

/* カラム間隔を少し広めに */
.css-ocqkz7, .egzxvld2 { gap: 2rem !important; }  /* 旧/新classの両対応（無視されてもOK） */
</style>
""", unsafe_allow_html=True)

# -----------------------------
# ヘッダー
# -----------------------------
st.title("聞いてポータル｜市長と議会のことばアーカイブ")
st.markdown(
    '<div class="small-muted">市長や議員の発言を検索・分析できるサイトです。政策やまちづくりに関する議論を、もっと身近に。</div>',
    unsafe_allow_html=True
)
st.divider()

# -----------------------------
# ナビゲーション用ヘルパー
# -----------------------------
def nav_button(label: str, page_py_path: str, key: str):
    """
    switch_page がある場合はページ遷移ボタン、
    ない場合は通常リンク（ボタン風テキスト）を描画。
    """
    if hasattr(st, "switch_page"):
        if st.button(label, key=key):
            st.switch_page(page_py_path)
    else:
        st.markdown(f"[{label} →](/?nav={page_py_path})")  # 代替の通常リンク（表示のみ）

# -----------------------------
# 3カード（市長／議員／まとめ）
# -----------------------------
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="kicker">👔 MAYOR</div>', unsafe_allow_html=True)
    st.markdown("### 聞いてミライ｜市長の発言を探す")
    st.write("施政方針や記者会見をRAGで検索。タグ・年度で絞り込み、要点要約で素早く把握できます。")
    nav_button("市長の発言を見る", APP_MAYOR_PATH, key="go_mayor")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="kicker">🏛 COUNCIL</div>', unsafe_allow_html=True)
    st.markdown("### 聞いてギカイ｜議員の発言を探す")
    st.write("会派・議員名・定例会で検索。質問と答弁のペア表示で、議論の流れが一目で分かります。")
    nav_button("議員の発言を見る", APP_COUNCIL_PATH, key="go_council")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="kicker">📊 INSIGHTS</div>', unsafe_allow_html=True)
    st.markdown("### 発言ダッシュボード｜ことばの傾向を知る")
    st.write("頻出ワード、共起ネットワーク、テーマの時系列推移、質問スタイル分析などを可視化。")
    nav_button("発言をまとめて見る", APP_SUMMARY_PATH, key="go_summary")
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# -----------------------------
# 概要 & 思い
# -----------------------------
st.subheader("このサイトについて")
st.write(
    "「聞いてポータル」は、**市長や議員の発言を検索・分析できるアーカイブ**です。"
    " 市政に関する議論をもっと身近に、もっと分かりやすく届けることを目指しています。\n\n"
    "- **市長の発言**（施政方針・記者会見など）\n"
    "- **議員の発言**（定例会での質問や答弁）\n"
    "- **発言まとめ**（頻出ワードや傾向分析）\n\n"
    "を収録し、検索や可視化を通じて「市政の見える化」に取り組んでいます。"
)

st.markdown("### 思い")
st.write(
    "政治やまちづくりの議論は、多くの人にとって「遠い存在」になりがちです。"
    " しかし、本来は私たちの暮らしに直結している大切なテーマ。\n\n"
    "このサイトを通じて、**誰でも市政の議論にアクセスしやすくなる環境**をつくり、"
    " 市民と行政との距離を少しでも縮めたいと考えています。"
)

st.markdown(f'🔗 **プロジェクト概要（Notion）**： [詳しくはこちら]({NOTION_URL})')

st.divider()

# -----------------------------
# 将来機能：地域名マッピング（Coming soon）
# -----------------------------
with st.expander("🗺️ 地域名マッピング（Coming soon）"):
    st.write(
        "発言に登場する地名（例：学校名、公園名、交差点など）を抽出し、地図上に可視化します。"
        " テーマ別の発言分布や、地域課題のホットスポット把握に役立てます。"
    )
    st.caption("※ 実装予定：地名抽出・ジオコーディング・地図描画（Streamlit/Leaflet・Kepler.gl 等）")

# -----------------------------
# フッター
# -----------------------------
st.markdown("---")
c1, c2, c3, c4 = st.columns([1,1,1,2])
with c1: st.write("© 2025 聞いてポータル")
with c2: st.write("Made with Streamlit")
with c3: st.write("Locale: Asia/Tokyo")
with c4:
    st.markdown(
        f'<span class="small-muted">最終更新: {datetime.now().strftime("%Y-%m-%d %H:%M")} JST</span>',
        unsafe_allow_html=True
    )
