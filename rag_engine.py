import os
import re
import yaml
import fitz  # PyMuPDF
import numpy as np
import faiss
import requests
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RagEngine:
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)

        # API Keys
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.serpapi_api_key = os.getenv("SERPAPI_API_KEY")

        if not self.gemini_api_key:
            print("Warning: GEMINI_API_KEY not found in environment.")

        # Models
        self.model_name = self.config.get("model", {}).get("name", "gemini-2.5-flash")
        self.embedder_model = self.config.get("embedder_model", "all-MiniLM-L6-v2")

        # Configure Gemini
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.model = genai.GenerativeModel(self.model_name)
        else:
            self.model = None

        # Initialize Embedder
        self.embedder = SentenceTransformer(self.embedder_model)

        # State
        self.pdf_chunks = []
        self.pdf_chunk_meta = []
        self.index_pdf = None
        self.documents = []

        # Parameters
        self.chunk_size = self.config.get("chunking", {}).get("chunk_size", 1100)
        self.chunk_overlap = self.config.get("chunking", {}).get("overlap", 150)
        self.retrieval_cfg = self.config.get("retrieval", {"top_k_pdf": 5, "top_k_web": 4, "max_chars": 2400})
        self.system_instructions = self.config.get("system_instructions", "You are a credit risk assistant.")

    def _load_config(self, config_path: str) -> dict:
        if config_path is None:
            # Look in current directory or same directory as this file
            possible_paths = ["config.yaml", os.path.join(os.path.dirname(__file__), "config.yaml")]
            for p in possible_paths:
                if os.path.exists(p):
                    config_path = p
                    break

        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def clean_text(self, s: str) -> str:
        s = s.replace("\u00a0", " ")
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip()

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            doc = fitz.open(pdf_path)
            text = []
            for page in doc:
                text.append(page.get_text())
            return "\n".join(text)
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return ""

    def chunk_text(self, text: str):
        chunks, i = [], 0
        while i < len(text):
            chunks.append(text[i:i+self.chunk_size])
            i += self.chunk_size - self.chunk_overlap
        return chunks

    def load_documents(self, pdf_folder: str = "./pdf_files/"):
        if not os.path.isdir(pdf_folder):
            print(f"PDF folder {pdf_folder} does not exist. Skipping PDF loading.")
            return

        pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
        for file in pdf_files:
            file_path = os.path.join(pdf_folder, file)
            raw = self.extract_text_from_pdf(file_path)
            raw = self.clean_text(raw)
            self.documents.append(raw)

            chunks = self.chunk_text(raw)
            for ci, c in enumerate(chunks):
                self.pdf_chunks.append(c)
                self.pdf_chunk_meta.append({"source": file, "chunk_id": ci})

        print(f"Loaded {len(pdf_files)} PDFs, created {len(self.pdf_chunks)} chunks.")

    def create_vector_store(self):
        if not self.pdf_chunks:
            print("No PDF chunks to index.")
            return

        embeddings = self.embedder.encode(self.pdf_chunks, convert_to_numpy=True, show_progress_bar=False)
        dimension = embeddings.shape[1]
        self.index_pdf = faiss.IndexFlatL2(dimension)
        self.index_pdf.add(embeddings.astype(np.float32))
        print("Vector store created successfully.")

    def web_search_serpapi(self, query: str, k: int = 4):
        if not self.serpapi_api_key:
            return []

        params = {
            "engine": "google",
            "q": query,
            "api_key": self.serpapi_api_key,
            "num": k,
            "hl": "en",
            "gl": "us", # default
        }

        try:
            r = requests.get("https://serpapi.com/search.json", params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            org = data.get("organic_results", []) or []
            cleaned = []
            for item in org:
                link = item.get("link", "")
                snippet = item.get("snippet", "")
                if link:
                    cleaned.append({
                        "title": item.get("title", ""),
                        "url": link,
                        "content": self.clean_text(snippet)[:2000]
                    })
            return cleaned
        except Exception as e:
            print(f"SerpAPI error: {e}")
            return []

    def retrieve_hybrid(self, query: str, top_k_pdf: int, top_k_web: int):
        pdf_hits = []
        if self.index_pdf is not None and self.pdf_chunks:
            q_emb = self.embedder.encode([query], convert_to_numpy=True).astype(np.float32)
            distances, indices = self.index_pdf.search(q_emb, min(top_k_pdf, len(self.pdf_chunks)))
            for rank, idx in enumerate(indices[0]):
                if idx == -1: continue # Should not happen unless k > n
                score = -float(distances[0][rank]) # FAISS L2 distance is smaller = better.
                pdf_hits.append({
                    "text": self.pdf_chunks[idx],
                    "score": score,
                    "source": self.pdf_chunk_meta[idx]["source"],
                    "chunk_id": self.pdf_chunk_meta[idx]["chunk_id"]
                })

        web_hits = []
        if top_k_web > 0:
            web_results = self.web_search_serpapi(query, k=8) # Fetch more, filter later if needed
            if web_results:
                # Re-rank web results using embeddings
                web_texts = [w["content"] for w in web_results if w.get("content")]
                if web_texts:
                    q_emb = self.embedder.encode([query], convert_to_numpy=True).astype(np.float32)
                    w_embs = self.embedder.encode(web_texts, convert_to_numpy=True).astype(np.float32)

                    # Cosine similarity
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

    def build_context(self, pdf_hits, web_hits, max_chars):
        combined = []
        # Interleave hits
        for i in range(max(len(pdf_hits), len(web_hits))):
            if i < len(pdf_hits): combined.append(("pdf", pdf_hits[i]))
            if i < len(web_hits): combined.append(("web", web_hits[i]))

        context_parts = []
        citations = []
        used = 0

        for typ, item in combined:
            snippet = item["text"].strip()
            if not snippet: continue

            header = f"[PDF:{item['source']}]" if typ == "pdf" else f"[WEB:{item.get('title','')}]"
            block = f"{header}\n{snippet}\n"

            if used + len(block) > max_chars:
                break

            context_parts.append(block)
            used += len(block)

            if typ == "web" and item.get("url"):
                citations.append(item["url"])

        return "\n\n---\n\n".join(context_parts), citations

    def query(self, user_question: str, context_data: str = None) -> str:
        if not self.model:
            return "Error: Google Gemini model not initialized (check API Key)."

        top_k_pdf = self.retrieval_cfg.get("top_k_pdf", 5)
        top_k_web = self.retrieval_cfg.get("top_k_web", 4)
        max_chars = self.retrieval_cfg.get("max_chars", 2400)

        pdf_hits, web_hits = self.retrieve_hybrid(user_question, top_k_pdf, top_k_web)
        retrieved_context, urls = self.build_context(pdf_hits, web_hits, max_chars)

        # Build prompt
        system_part = self.system_instructions

        context_data_part = ""
        if context_data:
            context_data_part = f"\n\nCURRENT DASHBOARD CONTEXT:\n{context_data}\n"

        final_prompt = (
            f"{system_part}\n"
            f"{context_data_part}\n"
            f"User question: {user_question}\n\n"
            f"Retrieved Context (grounding):\n"
            f"{'-'*40}\n{retrieved_context}\n{'-'*40}\n\n"
            "If you used any web evidence, add a short 'Sources:' list of URLs at the end."
        )

        try:
            response = self.model.generate_content(final_prompt)
            text = (response.text or "").strip()
            if urls:
                uniq_urls = list(dict.fromkeys(urls))
                text += "\n\nSources:\n" + "\n".join(uniq_urls)
            return text
        except Exception as e:
            return f"Error generating response: {e}"

if __name__ == "__main__":
    # Test
    engine = RagEngine()
    engine.load_documents()
    engine.create_vector_store()
    print("Test Query:", engine.query("What is the Gini coefficient?", context_data="Current Gini is 0.45"))
