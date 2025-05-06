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
def load_meta_json_from_drive(folder_id):
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
        return None

    file_id = files[0]["id"]
    request = drive_service.files().get_media(fileId=file_id)
    fh = BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    return json.load(fh)

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
meta_data = load_meta_json_from_drive(gdrive_folder_id)

if meta_data:
    # 絞り込みUI
    speakers = sorted(set(item.get("speaker") for item in meta_data if item.get("speaker")))
    types = sorted(set(item.get("type", "").strip() for item in meta_data if item.get("type")))
    speaker_filter = st.selectbox("発言者で絞り込み", [None] + speakers)
    type_filter = st.selectbox("発言種別で絞り込み", [None] + types)

    # ストップワード設定
    stop_words = ["こと", "ところ", "よう", "今回", "市", "もの", "中", "方", "ため"]  # 任意拡張OK

    freq = extract_keywords(meta_data, speaker_filter, type_filter, stop_words=stop_words)

    if freq:
        draw_wordcloud(freq)
    else:
        st.info("対象条件に一致する語句がありませんでした。")
