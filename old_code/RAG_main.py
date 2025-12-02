import os
import re
from typing import List, Dict, Tuple
import fitz  # PyMuPDF
import google.generativeai as genai
import numpy as np
import faiss
import requests
import gradio as gr
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import yaml

# create function that gets config.yaml parameters
def import_config_params(config_path: str = None) -> dict:
    """
    Load configuration from config.yaml located in the project folder.
    Returns a dict with configuration (empty dict if not found).
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            return cfg
    except FileNotFoundError:
        print(f"[config] Warning: config.yaml not found at {config_path}. Using defaults.")
        return {}
    except Exception as e:
        print(f"[config] Error loading config.yaml: {e}. Using defaults.")
        return {}

# -------------------- Setup --------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

# load YAML configuration and derive runtime parameters
config = import_config_params()
# model / embedder
MODEL_NAME = config.get("model", {}).get("name", "gemini-2.5-flash")
EMBEDDER_MODEL = config.get("embedder_model", "all-MiniLM-L6-v2")

# pdf + chunking settings
PDF_FOLDER = config.get("pdf", {}).get("folder", "./pdf_files/")
CHUNK_SIZE = config.get("chunking", {}).get("chunk_size", 1100)
CHUNK_OVERLAP = config.get("chunking", {}).get("overlap", 150)

# serpapi defaults
SERPAPI_DEFAULTS = config.get("serpapi", {"k": 8, "hl": "en", "gl": "es", "last_year_only": True})

# retrieval
RETRIEVAL = config.get("retrieval", {"top_k_pdf": 5, "top_k_web": 4, "max_chars": 2400})
TOP_K_PDF = RETRIEVAL.get("top_k_pdf", 5)
TOP_K_WEB = RETRIEVAL.get("top_k_web", 4)
MAX_CHARS = RETRIEVAL.get("max_chars", 2400)

SYSTEM_INSTRUCTIONS = config.get("system_instructions", (
    "You are a credit risk and regulation assistant. "
    "Use the provided context (PDF and web snippets) when relevant, "
    "and you may also use your own knowledge. "
    "When you use web snippets, include a 'Sources:' section with URLs."
))

GRADIO_CFG = config.get("gradio", {"server_name": "127.0.0.1", "server_port": 7860, "share": False})

# configure Gemini + embedder
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)
embedder = SentenceTransformer(EMBEDDER_MODEL)

def clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

# -------------------- PDF Ingestion --------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = []
    for page in doc:
        text.append(page.get_text())
    return "\n".join(text)

def chunk_text(text: str, chunk_size: int = 1100, overlap: int = 150):
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i+chunk_size])
        i += chunk_size - overlap
    return chunks

pdf_folder = PDF_FOLDER
if not os.path.isdir(pdf_folder):
    os.makedirs(pdf_folder, exist_ok=True)

documents, doc_sources = [], []
pdf_chunks, pdf_chunk_meta = [], []

pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
for file in pdf_files:
    file_path = os.path.join(pdf_folder, file)
    raw = extract_text_from_pdf(file_path)
    raw = clean_text(raw)
    documents.append(raw)
    doc_sources.append(file)

    chunks = chunk_text(raw, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    for ci, c in enumerate(chunks):
        pdf_chunks.append(c)
        pdf_chunk_meta.append({"source": file, "chunk_id": ci})

if pdf_chunks:
    pdf_embeddings = embedder.encode(pdf_chunks, convert_to_numpy=True, show_progress_bar=False)
    dimension = pdf_embeddings.shape[1]
    index_pdf = faiss.IndexFlatL2(dimension)
    index_pdf.add(pdf_embeddings.astype(np.float32))
else:
    pdf_embeddings = None
    index_pdf = None
    dimension = None

# -------------------- SerpAPI Search --------------------
def web_search_serpapi(query: str, k: int = None, hl: str = None, gl: str = None, last_year_only: bool = None):
    if not SERPAPI_API_KEY:
        return []

    k = k if k is not None else SERPAPI_DEFAULTS.get("k", 8)
    hl = hl if hl is not None else SERPAPI_DEFAULTS.get("hl", "en")
    gl = gl if gl is not None else SERPAPI_DEFAULTS.get("gl", "es")
    last_year_only = last_year_only if last_year_only is not None else SERPAPI_DEFAULTS.get("last_year_only", True)

    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": k,
        "hl": hl,
        "gl": gl,
        "safe": "active",
        "filter": "1",
    }
    if last_year_only:
        params["tbs"] = "qdr:y"

    try:
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        org = data.get("organic_results", []) or []
        cleaned = []
        for item in org:
            title = item.get("title", "")
            link = item.get("link", "") or item.get("url", "")
            snippet = item.get("snippet", "") or item.get("content", "")
            if not link:
                continue
            cleaned.append({
                "title": title,
                "url": link,
                "content": clean_text(snippet)[:2000]
            })
        return cleaned
    except Exception as e:
        print(f"[web] SerpAPI error: {e}")
        return []

def embed_texts(texts: List[str]):
    return embedder.encode(texts, convert_to_numpy=True)

# -------------------- Retrieval + Generation --------------------
def retrieve_hybrid(query: str, top_k_pdf: int = None, top_k_web: int = None):
    top_k_pdf = top_k_pdf if top_k_pdf is not None else TOP_K_PDF
    top_k_web = top_k_web if top_k_web is not None else TOP_K_WEB

    pdf_hits = []
    if index_pdf is not None and pdf_chunks:
        q_emb = embed_texts([query]).astype(np.float32)
        distances, indices = index_pdf.search(q_emb, min(top_k_pdf, len(pdf_chunks)))
        for rank, idx in enumerate(indices[0]):
            score = -float(distances[0][rank])
            pdf_hits.append({
                "text": pdf_chunks[idx],
                "score": score,
                "source": pdf_chunk_meta[idx]["source"],
                "chunk_id": pdf_chunk_meta[idx]["chunk_id"]
            })

    web_hits = []
    web_results = web_search_serpapi(query, k=8) if top_k_web > 0 else []
    if web_results:
        web_texts = [w["content"] for w in web_results if w.get("content")]
        if web_texts:
            q_emb = embed_texts([query]).astype(np.float32)
            w_embs = embed_texts(web_texts).astype(np.float32)
            qn = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
            wn = w_embs / (np.linalg.norm(w_embs, axis=1, keepdims=True) + 1e-12)
            sims = (wn @ qn[0])
            top_idx = np.argsort(-sims)[:top_k_web]
            for i in top_idx:
                w = web_results[i]
                web_hits.append({
                    "text": w["content"],
                    "score": float(sims[i]),
                    "url": w["url"],
                    "title": w.get("title", "")
                })
    return pdf_hits, web_hits

def build_context_and_citations(pdf_hits, web_hits, max_chars: int = 2400):
    combined = []
    for i in range(max(len(pdf_hits), len(web_hits))):
        if i < len(pdf_hits): combined.append(("pdf", pdf_hits[i]))
        if i < len(web_hits): combined.append(("web", web_hits[i]))

    context_parts, citations, used = [], [], 0
    for typ, item in combined:
        snippet = item["text"].strip()
        if not snippet:
            continue
        header = f"[PDF:{item['source']}#chunk{item['chunk_id']}]" if typ == "pdf" else f"[WEB:{item.get('title','')}]"
        block = f"{header}\n{snippet}\n"
        if used + len(block) > max_chars:
            break
        context_parts.append(block)
        used += len(block)
        if typ == "web" and item.get("url"):
            citations.append(item["url"])
    return "\n\n---\n\n".join(context_parts), citations

## Please add to  config.yaml file with all the parameters for the model configuration (web_search_serpapi params, SYSTEM_INSTRUCTIONS, etc.
def answer_query(user_input: str):
    if not user_input or not user_input.strip():
        return "", "Please enter a question."
    pdf_hits, web_hits = retrieve_hybrid(user_input, top_k_pdf=5, top_k_web=4)
    context, urls = build_context_and_citations(pdf_hits, web_hits, max_chars=2400)

    final_prompt = (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"User question: {user_input}\n\n"
        f"Context below. Prefer it for factual grounding.\n"
        f"{'-'*40}\n{context}\n{'-'*40}\n\n"
        "If you used any web evidence, add a short 'Sources:' list of URLs at the end."
    )

    response = model.generate_content(final_prompt)
    text = (response.text or "").strip()
    if urls:
        uniq_urls = list(dict.fromkeys(urls))
        text += "\n\nSources:\n" + "\n".join(uniq_urls)
    return text, ", ".join(dict.fromkeys(urls))

# -------------------- Gradio UI --------------------
with gr.Blocks(title="Capitalflow Hybrid RAG") as demo:
    gr.Markdown("## 🤖 Capitalflow Hybrid RAG (PDFs + Web via SerpAPI + Gemini 2.5 Flash)")

    with gr.Row():
        query = gr.Textbox(label="Ask a question", placeholder="e.g., Summarize PD calculation approaches...")
    with gr.Row():
        output = gr.Markdown(label="Answer")
    sources = gr.Textbox(label="Sources (URLs)", interactive=False)

    submit = gr.Button("Ask")
    submit.click(fn=answer_query, inputs=[query], outputs=[output, sources])

if __name__ == "__main__":
    demo.launch(
        server_name=GRADIO_CFG.get("server_name", "127.0.0.1"),
        server_port=GRADIO_CFG.get("server_port", 7860),
        share=bool(GRADIO_CFG.get("share", False))
    )
