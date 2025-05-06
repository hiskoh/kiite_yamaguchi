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

# --------------------------
# 日本語フォントの確保
# --------------------------
def get_font_path():
    font_dir = "./font"
    os.makedirs(font_dir, exist_ok=True)
    font_path = os.path.join(font_dir, "NotoSansCJKjp-Regular.otf")
    if not os.path.exists(font_path):
        url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Japanese/NotoSansCJKjp-Regular.otf"
        urllib.request.urlretrieve(url, font_path)
    return font_path

# -------------------------------
# Google Driveからmeta.json取得
# -------------------------------
def load_meta_json_from_drive(folder_id):
    creds = Credentials.from_service_account_info(
        st.secrets["gsheets_service_account"],
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    drive_service = build("drive", "v3", credentials=creds)

    results = drive_service.files().list(
        q=f"'{folder_id}' in parents and name contains '.meta.json' and trashed=false",
        spaces='drive',
        fields='files(id, name)',
    ).execute()

    files = results.get("files", [])
    if not files:
        st.warning("meta.json が見つかりませんでした")
        return None

    file_id = files[0]['id']
    request = drive_service.files().get_media(fileId=file_id)
    fh = BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    return json.load(fh)

# ------------------------
# 形態素解析＋ワード集計
# ------------------------
def extract_noun_frequencies(meta_data, speaker_filter=None, type_filter=None):
    tagger = Tagger()
    counter = Counter()

    for item in meta_data:
        if speaker_filter and item.get("speaker") != speaker_filter:
            continue
        if type_filter and item.get("type") != type_filter:
            continue

        text = item.get("text", "")
        words = [w.surface for w in tagger(text) if w.feature.pos1 == "名詞" and len(w.surface) > 1]
        counter.update(words)

    return counter

# ------------------
# ワードクラウド描画
# ------------------
def draw_wordcloud(freq):
    font_path = get_font_path()
    wc = WordCloud(font_path=font_path, width=800, height=400, background_color="white")
    wc.generate_from_frequencies(freq)

    buf = BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(buf, format="png")
    st.image(buf)

# -----------------
# Streamlit UI
# -----------------
st.title("きいてミライ - ワードクラウド生成")

gdrive_folder_id = st.secrets["kiite-mirai"]["GOOGLE_MIRAI_DATA_ID"]
meta_data = load_meta_json_from_drive(gdrive_folder_id)

if meta_data:
    speakers = sorted(set(item["speaker"] for item in meta_data if item.get("speaker")))
    types = sorted(set(item["type"] for item in meta_data if item.get("type")))

    speaker_filter = st.selectbox("発言者で絞り込み（任意）", [None] + speakers)
    type_filter = st.selectbox("発言種別で絞り込み（任意）", [None] + types)

    freq = extract_noun_frequencies(meta_data, speaker_filter, type_filter)

    if freq:
        draw_wordcloud(freq)
    else:
        st.info("対象条件に一致する語句がありませんでした。")
