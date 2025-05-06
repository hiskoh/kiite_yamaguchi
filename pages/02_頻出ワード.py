import streamlit as st
import json
from wordcloud import WordCloud
from fugashi import Tagger
from collections import Counter
import matplotlib.pyplot as plt
from io import BytesIO
import os
import urllib.request
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ------------------------
# 日本語フォントのダウンロード
# ------------------------
def get_font_path():
    font_dir = "./font"
    os.makedirs(font_dir, exist_ok=True)
    font_path = os.path.join(font_dir, "NotoSansCJKjp-Regular.otf")
    if not os.path.exists(font_path):
        url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf"
        urllib.request.urlretrieve(url, font_path)
    return font_path

# ------------------------
# Google Driveからmeta.jsonを読み込む
# ------------------------
def load_all_meta_jsons_from_drive(folder_id):
    creds = Credentials.from_service_account_info(
        st.secrets["gsheets_service_account"],
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    drive_service = build("drive", "v3", credentials=creds)

    results = drive_service.files().list(
        q=f"'{folder_id}' in parents and name contains '.meta.json' and trashed=false",
        spaces='drive',
        fields='files(id, name)'
    ).execute()

    files = results.get("files", [])
    if not files:
        st.warning("meta.json が見つかりませんでした")
        return []

    all_meta = []
    for f in files:
        file_id = f["id"]
        request = drive_service.files().get_media(fileId=file_id)
        fh = BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        fh.seek(0)
        all_meta.extend(json.load(fh))  # listとして追加

    return all_meta


# ------------------------
# 名詞抽出＋頻度カウント
# ------------------------
def extract_keywords(meta_data, speaker_filter=None, type_filter=None, min_length=2, stop_words=None):
    tagger = Tagger()
    counter = Counter()
    stop_words = stop_words or []

    for item in meta_data:
        if speaker_filter and item.get("speaker") != speaker_filter:
            continue
        if type_filter and item.get("type", "").strip() != type_filter:
            continue

        text = item.get("text", "")
        words = [
            w.surface for w in tagger(text)
            if w.feature.pos1 == "名詞"
            and len(w.surface) >= min_length
            and w.surface not in stop_words
        ]
        counter.update(words)

    return counter

# ------------------------
# ワードクラウド描画（streamlitにそのまま）
# ------------------------
def draw_wordcloud(freq):
    font_path = get_font_path()
    wc = WordCloud(
        font_path=font_path,
        width=800,
        height=400,
        background_color="white",
        colormap="tab10", 
        regexp=r"[\w']+"
    )
    wc.generate_from_frequencies(freq)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# ------------------------
# Streamlit 本体
# ------------------------
st.title("きいてミライ：ワードクラウド")

# secrets.toml から固定フォルダIDを取得
gdrive_folder_id = st.secrets["kiite-mirai"]["GOOGLE_MIRAI_DATA_ID"]
meta_data = load_all_meta_jsons_from_drive(gdrive_folder_id)

if meta_data:
    # 絞り込みUI（"ALL"を表示ラベル、Noneを内部値にマッピング）
    speakers = sorted(set(item.get("speaker") for item in meta_data if item.get("speaker")))
    types = sorted(set(item.get("type", "").strip() for item in meta_data if item.get("type")))

    speaker_label = st.selectbox("発言者で絞り込み", ["ALL"] + speakers)
    type_label = st.selectbox("発言種別で絞り込み", ["ALL"] + types)

    speaker_filter = None if speaker_label == "ALL" else speaker_label
    type_filter = None if type_label == "ALL" else type_label


    # ワードクラウドに使わない文字列を指定（引用元：https://github.com/yukihoz/chuoku_gijiroku/blob/master/gijiroku_streamlit.py）
    stop_words = ["日本","山口","市民","予算","整備","ページ","制度","対策","づくり","連携","推進","促進","はじめ","活用","予定","機能","強化","実施","採択","計画","補助","提案","関連","目的","視点","視点","認識","取組","辺り","具体","面","令和","様","辺","なし","分","款","皆","さん","議会","文","場所","現在","ら","方々","こちら","性","化","場合","対象","一方","皆様","考え","それぞれ","意味","とも","内容","とおり","目","事業","つ","見解","検討","本当","議論","民","地域","万","確認","実際","先ほど","前","後","利用","説明","次","あたり","部分","状況","わけ","話","答弁","資料","半ば","とき","支援","形","今回","中","対応","必要","今後","質問","取り組み","終了","暫時","午前","たち","九十","八十","七十","六十","五十","四十","三十","問題","提出","進行","付託","議案","動議","以上","程度","異議","開会","午後","者","賛成","投票","再開","休憩","質疑","ただいま","議事","号","二十","平成","等","会","日","月","年","年度","委員","点","区","委員会","賛成者","今","もの","こと","ふう","ところ","ほう","これ","私","わたし","僕","あなた","みんな","ただ","ほか","それ", "もの", "これ", "ところ","ため","うち","ここ","そう","どこ", "つもり", "いつ","あと","もん","はず","こと","そこ","あれ","なに","傍点","まま","事","人","方","何","時","一","二","三","四","五","六","七","八","九","十"]

    freq = extract_keywords(meta_data, speaker_filter, type_filter, stop_words=stop_words)

    if freq:
        draw_wordcloud(freq)
    else:
        st.info("対象条件に一致する語句がありませんでした。")


    st.markdown("※ 本ワードクラウドは形態素解析（名詞抽出）を行い、以下の一般的な語句を除外しています。")
    stop_list_text = "、".join(stop_words)
    with st.expander("除外している語句一覧を見る"):
        st.markdown(stop_list_text)
