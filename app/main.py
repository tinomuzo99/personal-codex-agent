from __future__ import annotations
import os

# --- path bootstrap (top of app/main.py) ---
import os, sys, pathlib

# Ensure the repo root is on sys.path (so "from app import ..." works)
THIS_FILE = pathlib.Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent  # one level up from /app
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Prefer absolute import; fall back to relative if needed
try:
    from app.rag import RAGIndex
    from app.modes import MODES
    from app.voice import PERSONA
except Exception:
    # When executed in certain contexts that treat main.py as part of the package
    from .rag import RAGIndex
    from .modes import MODES
    from .voice import PERSONA
    
import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict, Any
from openai import OpenAI
from app.rag import RAGIndex
from app.modes import MODES
from app.voice import PERSONA
import re
# --- path bootstrap (top of app/main.py) ---
import os, sys, pathlib

# Ensure the repo root is on sys.path (so "from app import ..." works)
THIS_FILE = pathlib.Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent  # one level up from /app
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Prefer absolute import; fall back to relative if needed
try:
    from app.rag import RAGIndex
    from app.modes import MODES
    from app.voice import PERSONA
except Exception:
    # When executed in certain contexts that treat main.py as part of the package
    from .rag import RAGIndex
    from .modes import MODES
    from .voice import PERSONA

load_dotenv()

# Use Streamlit secrets if env vars aren’t present
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
try:
    import streamlit as st
    OPENAI_API_KEY = OPENAI_API_KEY or st.secrets.get("OPENAI_API_KEY")
except Exception:
    pass

st.set_page_config(page_title="Ubundi Personal Codex Agent", page_icon="🗂️", layout="wide")

# Sidebar controls
with st.sidebar:
    st.markdown("## Settings")
    mode_name = st.selectbox("Mode", list(MODES.keys()), index=0)
    rag_enabled = st.toggle("Use RAG (retrieval)", value=os.environ.get("RAG_ENABLED", "true").lower() == "true")
    top_k = st.slider("Top‑k passages", 1, 10, int(os.environ.get("TOP_K", 5)))
    temperature = st.slider("Temperature", 0.0, 1.0, float(os.environ.get("TEMPERATURE", 0.5)))
    model_name = st.text_input("Model name", os.environ.get("MODEL_NAME", "gpt-4o-mini"))
    st.caption("Switch modes to change style; toggle RAG for citations.")


st.title("Ubundi Personal Codex Agent")

# Ensure index availability (lazy)
@st.cache_resource(show_spinner=False)
def load_index() -> RAGIndex:
    idx = RAGIndex()
    try:
        idx.load()
    except Exception as e:
        st.info("Index not found or failed to load. Upload docs to `data/raw/` and run `make reindex`.")
        raise e
    return idx

# Simple chat state
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: {role, content}



def to_first_person(text: str) -> str:
    rules = [
        (r"\bTinomutendayi Muzondidya is\b", "I am"),
        (r"\bTinomutendayi is\b", "I am"),
        (r"\bMuzondidya\b", "I"),
        (r"\bhe has\b", "I have"),
        (r"\bhe is\b", "I am"),
        (r"\bhis\b", "my"),
        (r"\bHis\b", "My"),
        (r"\bHe has\b", "I have"),
        (r"\bHe is\b", "I am"),
    ]
    out = text
    out = out.replace("Tmy", "My")
    out = out.replace("tmy", "my")
    for pat, repl in rules:
        out = re.sub(pat, repl, out)
    return out

def llm_respond(system_prompt: str, user_prompt: str, temperature: float, model: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")  or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return "⚠️ OPENAI_API_KEY not set. Please configure your .env file."

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content or ""


with st.container(border=True):
    q = st.chat_input("Ask about Tinomutendayi…")
    if q:
        st.session_state.messages.append({"role": "user", "content": q})


# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


# On new user input, answer
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_q = st.session_state.messages[-1]["content"]
    mode = MODES[mode_name]

    retrieved: List[Dict[str, Any]] = []
    citations = ""

    if rag_enabled:
        try:
            idx = load_index()
            retrieved = idx.retrieve(last_q, k=top_k, rerank=False)
            if retrieved:
                preview = []
                for r in retrieved:
                    head = f"{r['cite_id']} {os.path.basename(r.get('source_name') or r['source'])} · chunk {r['chunk_id']}"
                    body = r["text"].replace("\n", " ").strip()
                    if len(body) > 300:
                        body = body[:300] + "…"
                    preview.append(f"**{head}**\n\n> {body}")
                citations = "\n\n---\n\n".join(preview)
        except Exception as e:
            st.warning(f"Retrieval failed: {e}")

    # Build prompts
    sys_prompt = (
        f"{mode.system}\n\n"
        f"{PERSONA}\n\n"
        "Always answer in the **first person** ('I'), never refer to Tinomutendayi in the third person. "
        "Do not prefix sentences with my name. Write as if I am directly speaking about myself."
)

    user_prompt = last_q

    if rag_enabled and retrieved:
        ctx = "\n\n".join(f"{r['cite_id']} {r['text']}" for r in retrieved)
        user_prompt = (
            f"Question: {last_q}\n\n"
            f"Use the following context if relevant. Cite using the bracketed ids [#].\n\n{ctx}\n\n"
            f"Style hint: {mode.style_hint}"
        )
    else:
        user_prompt = (
            f"Question: {last_q}\n\n"
            f"No reliable sources were retrieved. Answer briefly and cautiously, and explicitly say so. "
            f"Style hint: {mode.style_hint}."
        )

    def enforce_consistency(text: str) -> str:
        # If both "I " and "he " appear, prefer first person
        if " he " in text and " I " in text:
            text = text.replace(" he ", " I ")
        return text

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer = llm_respond(
                sys_prompt, user_prompt, temperature=temperature, model=model_name
            )
            if mode_name in ["Interview", "Personal storytelling", "Humble brag"]:
                answer = to_first_person(answer)
                answer = enforce_consistency(answer)
            if answer.strip():
                st.markdown(answer)
                if citations:
                    with st.expander("Retrieved context & citations"):
                        st.markdown(citations)
            else:
                st.markdown("(No response)")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
