# Ubundi Personal Codex Agent

A lightweight, context-aware agent that answers questions about **Tinomutendayi Muzondidya** using retrieval-augmented generation (RAG) over curated personal documents. It offers multiple conversation modes (Interview, Personal Storytelling, Fast Facts, Humble Brag) and a minimal Streamlit UI.

---

## 1) System Setup & Design Choices

### Tech stack
- **Frontend:** Streamlit chat UI (sidebar for mode, temperature, RAG on/off).
- **RAG:** Sentence-Transformers embeddings + FAISS vector index.
- **Chunking:** ~900 tokens per chunk with ~180-token overlap for stable retrieval.
- **Model(s):** `sentence-transformers/all-MiniLM-L6-v2` (embeddings); OpenAI chat model for answers.
- **Citations:** Inline bracketed IDs like `[1]`, with an expander showing retrieved snippets and source file names.

### Directory structure
```
personal-codex/
├─ app/
│  ├─ main.py          # Streamlit UI & Chat loop
│  ├─ rag.py           # Ingestion, indexing, and retrieval
│  ├─ modes.py         # Conversation modes (style + system instructions)
│  ├─ voice.py         # Persona/voice guidance
│  └─ utils.py         # Loaders, chunking, helpers
├─ data/
│  ├─ raw/             # (local) CV + supporting docs (PDF/MD/TXT)  [gitignored]
│  └─ index/           # (local) FAISS index + metadata             [gitignored]
├─ tests/
│  └─ fixtures/
├─ .env.example        # example config (safe to commit)
├─ requirements.txt
├─ Makefile            # optional shortcuts (setup / reindex / run)
└─ README.md
```

### Key design choices
- **Small, fast embeddings** (`all-MiniLM-L6-v2`) for quick indexing and low latency.
- **Cosine similarity** with FAISS `IndexFlatIP` over normalised vectors.
- **First-person voice** enforcement using a strong system prompt plus a light post-processor to avoid third-person drift.
- **Safety behaviour:** If retrieval confidence is low (no context), the assistant answers cautiously and signals limited sources.

### Environment variables
Create a local `.env` (not committed) by copying `.env.example`:
```
OPENAI_API_KEY=...
MODEL_NAME=gpt-4o-mini
EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2
TOP_K=5
TEMPERATURE=0.5
RAG_ENABLED=true
```

---

## 2) Getting Started

### Prerequisites
- Python 3.10+ recommended
- Windows, macOS, or Linux
- Git (for cloning/pushing to GitHub)

### Install & Run (Windows-friendly)
```powershell
# 1) Create venv & activate
python -m venv .venv
.\.venv\Scripts\activate

# 2) Install dependencies
python -m pip install -r requirements.txt

# 3) Prepare data
# Put your CV + 2–3 supporting docs into: data\raw\  (PDF/MD/TXT)

# 4) Build the index
python -m app.rag --reindex

# 5) Run the app (module form avoids import issues)
python -m streamlit run app/main.py
```

If Windows Defender blocks executables, run Streamlit via Python module:
```powershell
.\.venv\Scripts\python.exe -m streamlit run app\main.py --server.port 8502 --server.headless true
```

---

## 3) Sample Questions & Expected Answers

- **“What kind of engineer are you?”**  
  *Expected:* A first-person summary
  *Example:* I am a Chemical Engineer with a strong focus on waste valorisation and sustainable practices. My final year project involved assessing the kinetics of steam distillation to extract D-Limonene from citrus peel waste, which sparked my interest in obtaining high-value products from waste materials  `[1] `. Currently, I am pursuing a Master of Engineering in Industrial Engineering with a focus on Data Science, where I am applying advanced data analysis techniques to improve recommender systems  `[2] `.

- **“What are your strongest technical skills?”**  
  *Expected:* A first-person summary of key technical skills
  *Example:* My strongest technical skills include programming in Python, R, and SQL, which I utilise for data analysis and machine learning projects. I have proficiency in data visualisation tools such as Tableau and R-Studio, and I am adept at data cleaning and transformation. My engineering background equips me with skills in process design and experimental design, particularly in chemical engineering contexts. Additionally, I have experience with deep learning frameworks like PyTorch, demonstrated through my work on convolutional neural networks for artwork classification and transformer models for analysing unstructured user feedback `[1]`.

- **“What projects or experiences are you most proud of?”**  
  *Expected:* 1–2 concise project stories.
  *Example:* I take great pride in my final year project at Stellenbosch University, where I assessed the kinetics of steam distillation to extract D-limonene from citrus peel waste. This project ignited my passion for waste valorisation and the potential to derive high-value products from waste materials. I developed my research skills significantly under the guidance of my supervisor, Mr. Zwonaka Mapholi, and I am grateful for the support I received from the Department of Chemical Engineering `[1][2]`.
 
  Additionally, I am proud of my work in optimising the extraction process through ultrasound-aided steam distillation, which addresses some of the traditional method's limitations, such as longer extraction times and energy costs `[1][3]`. My involvement in these projects has solidified my interest in pursuing innovative solutions in chemical engineering and data science.

- **“What do you value in a team or company culture?”**  
  *Expected:* First-person reflection on collaboration, inclusivity, ethical practice, and continuous learning; brief, professional tone.
  *Example:* I value a team or company culture that prioritises collaboration, open communication, and continuous learning. A supportive environment that encourages innovation and embraces diverse perspectives is essential for fostering creativity and problem-solving. I also appreciate a culture that recognises and rewards individual contributions while maintaining a strong sense of community and shared goals. Ultimately, I believe that a positive culture enhances productivity and drives success for both individuals and the organisation as a whole.

- **“What’s your approach to learning or debugging something new?”**  
  *Expected:* A short, stepwise approach referencing an example.
  *Example:* My approach to learning or debugging something new involves a structured, iterative process. I begin by defining the problem clearly and breaking it down into manageable parts. For example, during my final year project on steam distillation, I faced challenges with pressure build-up and cooling rates. I systematically tested different solutions, such as adjusting sieve sizes and improving the cooling system, while documenting the outcomes to understand their effects on essential oil yield  `[1][4] `.
  I also leverage my existing knowledge and data analysis skills to inform my decisions. In a project involving Fantasy Premier League, I applied data-driven techniques to optimise team selection, iterating through various prompts and constraints to refine my approach based on performance metrics  ` [1]`. This method of trial and error, combined with analytical thinking, helps me to learn effectively and adapt my strategies to achieve better results.
 

> Answers include a “Retrieved context & citations” expander with the exact snippets and source files used.

---

## 4) What I’d Improve With More Time

- Add a simple file uploader in the sidebar to ingest new docs without touching the filesystem.
- Streamlit Cloud or Dockerfile + Render/Fly; secrets via environment variables, not in repo.

---




