import streamlit as st
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials

# ✅ ページ設定
st.set_page_config(page_title="きいてみらい山口", layout="wide")
st.title("きいてみらい山口について")

# ✅ 本文
st.markdown("""
このプロジェクトは、山口市議会の議事録をもとに、市民が議会の議論に簡単にアクセスできるようにするための取り組みです。

データは本サイトの制作者がクラウド上に保存した議事録ファイルを元にしています。現在、検索対象としている議事録は下記の通りです。
""")

# ✅ 出典一覧の取得関数
def get_drive_service():
    creds = Credentials.from_service_account_info(
        st.secrets["gsheets_service_account"],
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)

def list_index_sources(folder_id, service):
    sources = set()
    folders_to_search = [folder_id]

    while folders_to_search:
        current_folder_id = folders_to_search.pop()
        query = f"'{current_folder_id}' in parents and trashed = false"
        response = service.files().list(q=query, fields="files(id, name, mimeType, parents)").execute()

        for file in response.get("files", []):
            if file["mimeType"] == "application/vnd.google-apps.folder":
                folders_to_search.append(file["id"])
            elif file["name"].endswith(".index"):
                parent_id = file["parents"][0]
                parent = service.files().get(fileId=parent_id, fields="name").execute()
                folder_name = parent["name"]
                base_name = file.get("name", "").split("/")[-1].removesuffix(".index") if file.get("name", "").endswith(".index") else ""
                sources.add(f"{folder_name}/{base_name}")
    return sorted(sources)

# ✅ Drive接続・出典一覧表示
try:
    folder_id = st.secrets["kiite-gikai"]["GOOGLE_GIKAI_DATA_ID"]
    drive_service = get_drive_service()
    source_list = list_index_sources(folder_id, drive_service)

    with st.expander("📂 議事録の出典一覧（クリックで表示）", expanded=False):
        if source_list:
            for path in source_list:
                st.markdown(f"- {path}")
        else:
            st.markdown("（出典が見つかりませんでした）")

except Exception as e:
    st.warning(f"⚠️ 出典一覧を取得できませんでした: {e}")

# ✅ フッター
st.caption("""
⚠️ 回答は生成AIによるものであり、正確性を保証するものではありません。  
🙌 本プロジェクトは個人により運営されています。ご支援いただける方はぜひこちらから：  
[💛 codocで支援する](https://codoc.jp/sites/p8cEFlTZQA/entries/MMZnODc1dw)
""")
