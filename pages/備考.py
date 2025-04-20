import streamlit as st
from openai import OpenAI
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import random

# ✅ ページ設定
st.set_page_config(page_title="きいてみらい山口", layout="wide")
st.title("きいてみらい山口について")

# ✅ 説明セクション
st.caption("""
⚠️ 回答は生成AIによるものであり、正確性を保証するものではありません。  
🙌 本プロジェクトは個人により運営されています。ご支援いただける方はぜひこちらから：  
[💛 codocで支援する](https://codoc.jp/sites/p8cEFlTZQA/entries/MMZnODc1dw)
""")

# ✅ 出典取得関数（Drive API 事前に service = build('drive', 'v3') 済前提）
def list_txt_sources(folder_id, service):
    sources = set()
    folders_to_search = [folder_id]

    while folders_to_search:
        current_folder_id = folders_to_search.pop()
        query = f"'{current_folder_id}' in parents and trashed = false"
        response = service.files().list(q=query, fields="files(id, name, mimeType, parents)").execute()

        for file in response.get("files", []):
            if file["mimeType"] == "application/vnd.google-apps.folder":
                folders_to_search.append(file["id"])
            elif file["mimeType"] == "text/plain" and file["name"].endswith(".txt"):
                parent_id = file["parents"][0]
                parent = service.files().get(fileId=parent_id, fields="name").execute()
                folder_name = parent["name"]
                base_name = file["name"].replace(".txt", "")
                full_path = f"{folder_name}/{base_name}"
                sources.add(full_path)
    return sorted(sources)

# ✅ 出典一覧（Drive IDとDrive APIがある場合のみ）
try:
    from googleapiclient.discovery import build
    from google.colab import auth
    import os
    auth.authenticate_user()
    service = build('drive', 'v3')

    INPUT_FOLDER_ID = "ここにDriveのフォルダIDを入力"

    source_list = list_txt_sources(INPUT_FOLDER_ID, service)

    with st.expander("📂 議事録の出典一覧を表示", expanded=False):
        for path in source_list:
            st.markdown(f"- {path}")

except Exception as e:
    st.warning("⚠️ 出典一覧を取得できませんでした（Drive API接続が必要です）")
