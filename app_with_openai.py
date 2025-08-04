
import os, tempfile, uuid
from pathlib import Path
from datetime import datetime
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from streamlit.runtime.uploaded_file_manager import UploadedFile


st.set_page_config(
    page_title="Baza wiedzy PKP",
    page_icon="https://cdn-icons-png.flaticon.com/512/864/864685.png",
)

CONFIG_PATH = Path("config.yaml")
CHROMA_PATH = Path("./rag-chroma")
UPLOAD_DIR = Path("uploads"); UPLOAD_DIR.mkdir(exist_ok=True)

load_dotenv()

def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f, Loader=SafeLoader)

def _save_config(cfg: dict, path: Path):
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)

config = _load_config(CONFIG_PATH)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or config.get("openai_api_key")
if not OPENAI_API_KEY:
    st.error("Brak klucza OPENAI_API_KEY w .env / środowisku / config.yaml")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

COOKIE_NAME = config["cookie"]["name"]
if COOKIE_NAME in st.session_state and "name" not in st.session_state[COOKIE_NAME]:
    del st.session_state[COOKIE_NAME]

authenticator = stauth.Authenticate(
    credentials=config["credentials"],
    cookie_name=COOKIE_NAME,
    key=config["cookie"]["key"],
    cookie_expiry_days=config["cookie"]["expiry_days"],
)

name, auth_status, username = authenticator.login(location="main")
if auth_status is False:
    st.error("❌ Nieprawidłowy login lub hasło."); st.stop()
elif auth_status is None:
    st.warning("🔐 Wprowadź dane logowania."); st.stop()

role = config["credentials"]["usernames"][username].get("role", "viewer").lower()

if "session_uid" not in st.session_state:
    st.session_state["session_uid"] = str(uuid.uuid4())[:8]

CHAT_KEY = f"messages_{username}_{st.session_state['session_uid']}"
VSTORE_KEY = "vector_store_ready"

st.session_state.setdefault(CHAT_KEY, [])
st.session_state.setdefault(VSTORE_KEY, CHROMA_PATH.exists() and any(CHROMA_PATH.iterdir()))


TENANT = "default_tenant"; DATABASE = "default_database"

def _chromadb_client():
    return chromadb.PersistentClient(path=str(CHROMA_PATH), tenant=TENANT, database=DATABASE, settings=Settings(allow_reset=True))

def _collection():
    return _chromadb_client().get_or_create_collection(
        name="PKP",
        embedding_function=OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name="text-embedding-3-small"),
        metadata={"hnsw:space": "cosine"},
    )


def _process_document(uploaded: UploadedFile):
    tmp = tempfile.mktemp(suffix=".pdf")
    with open(tmp, "wb") as f: f.write(uploaded.read())
    pages = PyMuPDFLoader(tmp).load(); os.remove(tmp)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400, separators=["\n\n", "\n", ".", "?", "!", " "])
    text_chunks = splitter.split_text("\n".join(p.page_content for p in pages))
    return [Document(page_content=t, metadata={"source": uploaded.name}) for t in text_chunks]

def _add_to_store(docs: list[Document], tag: str):
    col = _collection()
    col.upsert(
        documents=[d.page_content for d in docs],
        metadatas=[d.metadata for d in docs],
        ids=[f"{tag}_{i}" for i in range(len(docs))],
    )
    st.session_state[VSTORE_KEY] = True
    


def _search(q: str, top_n=5):
    return _collection().query(query_texts=[q], n_results=top_n).get("documents", [[]])[0]

def _context(question: str, docs: list[str], k=3):
    ranks = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2").rank(question, docs, top_k=k)
    return "\n".join(docs[r["corpus_id"]] for r in ranks)

def _answer(ctx: str, q: str):
    prompt = f"Kontekst:\n{ctx}\n\nPytanie: {q}"
    stream = client.chat.completions.create(
        model="gpt-4.1-nano-2025-04-14",
        stream=True,
        messages=[{"role":"system","content":"Odpowiadasz wyłącznie na podstawie kontekstu."},{"role":"user","content":prompt}],
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content

def _uploader():
    
    with st.sidebar:
        st.markdown("### Prześlij PDF do bazy")
        pdfs = st.file_uploader("Wybierz pliki PDF", type=["pdf"], accept_multiple_files=True)
        if st.button("Prześlij") and pdfs:
            status = st.empty()  
            total = len(pdfs)
            for idx, pdf in enumerate(pdfs, start=1):
                try:
                    _add_to_store(_process_document(pdf), pdf.name.replace(" ", "_").replace("-", "_"))
                    remaining = total - idx
                    status.success(f"✅ Dodano: {pdf.name} | Pozostało: {remaining}")
                except Exception as e:
                    status.error(f"❌ {pdf.name}: {e}")
            st.toast("Import zakończony ✔️")
            st.session_state[VSTORE_KEY] = True
        if st.button("Wyczyść bazę"):
            _chromadb_client().reset(); st.session_state[VSTORE_KEY]=False; st.success("Baza wyczyszczona")

def _chat():
    st.title("📚 PKP – Baza wiedzy")
    for m in st.session_state[CHAT_KEY]:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if prompt := st.chat_input("Zadaj pytanie…"):
        st.session_state[CHAT_KEY].append({"role":"user","content":prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            placeholder = st.empty()
            if not st.session_state[VSTORE_KEY]:
                ans = "⚠️ Baza jest pusta. Dodaj PDF."
                placeholder.markdown(ans)
            else:
                with st.spinner("Szukanie odpowiedzi…"):
                    try:
                        docs = _search(prompt)
                        if not docs:
                            ans = "Brak dopasowań."
                        else:
                            ctx = _context(prompt, docs)
                            ans = "";  
                            for part in _answer(ctx, prompt):
                                ans += part; placeholder.markdown(ans)
                    except Exception as e:
                        ans = f"Błąd: {e}"; placeholder.markdown(ans)
            st.session_state[CHAT_KEY].append({"role":"assistant","content":ans})


def _hash(p: str): return stauth.Hasher([p]).generate()[0]

def _admin():
    st.title("🛠️ Panel administracyjny")
    users = config["credentials"]["usernames"]

   
    st.subheader("Aktualni użytkownicy")
    cols = st.columns((3,3,4,2,1))
    for c,h in zip(cols,["Login","Imię i nazwisko","E-mail","Rola",""]): c.markdown(f"**{h}**")
    for uname,data in users.items():
        c1,c2,c3,c4,c5 = st.columns((3,3,4,2,1))
        c1.write(uname); c2.write(data.get("name","")); c3.write(data.get("email","")); c4.write(data.get("role",""))
        if c5.button("🗑️", key=f"del_{uname}"):
            if uname==username: st.error("Nie możesz usunąć własnego konta.")
            else:
                users.pop(uname); _save_config(config, CONFIG_PATH); st.success("Usunięto użytkownika"); st.rerun()
        with c1.expander("🔑 Zmień hasło"):
            p1 = st.text_input("Nowe hasło", type="password", key=f"p1_{uname}")
            p2 = st.text_input("Powtórz", type="password", key=f"p2_{uname}")
            if st.button("Zapisz", key=f"chg_{uname}"):
                if not p1 or p1!=p2: st.error("Hasła puste lub niezgodne")
                else:
                    users[uname]["password"]=_hash(p1); _save_config(config, CONFIG_PATH); st.success("Zmieniono hasło"); st.rerun()

    st.markdown("---")
    
    st.subheader("Dodaj użytkownika")
    with st.form("add_user", clear_on_submit=True):
        lu, ln, le = st.text_input("Login *"), st.text_input("Imię i nazwisko *"), st.text_input("E-mail")
        role_new = st.selectbox("Rola", ["uploader","consultant","admin"])
        p1 = st.text_input("Hasło *", type="password"); p2 = st.text_input("Powtórz hasło *", type="password")
        if st.form_submit_button("Dodaj"):
            if lu in users: st.error("Login istnieje")
            elif "" in (lu,ln,p1,p2): st.error("Pola * są wymagane")
            elif p1!=p2: st.error("Hasła różne")
            else:
                users[lu] = {"name":ln,"email":le,"role":role_new,"password":_hash(p1),"created":datetime.now().strftime("%Y-%m-%d %H:%M")}
                _save_config(config, CONFIG_PATH); st.success("Dodano użytkownika"); st.rerun()


def main():
    with st.sidebar: st.markdown(f"# :blue[{name}]")
    if role=="admin":
        _admin(); _uploader()
    elif role=="uploader":
        _uploader(); _chat()
    else:
        st.sidebar.info("Masz dostęp tylko do czatu."); _chat()
    with st.sidebar:
        authenticator.logout(':red[Wyloguj się]')

if __name__ == "__main__":
    main()
