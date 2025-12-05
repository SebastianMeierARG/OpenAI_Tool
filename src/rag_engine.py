import os
import yaml
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

class RagEngine:
    """
    RAG Engine for "Chat with your Data & Docs".
    Uses manual retrieval and f-strings to avoid complex Chain dependencies.
    """
    def __init__(self, config_path="config.yaml"):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        # Load config safely
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        # Fallbacks for config keys
        rag_config = self.config.get('rag', {})
        self.docs_folder = rag_config.get('documents_folder', './documents')
        self.chunk_size = rag_config.get('chunk_size', 1000)
        self.chunk_overlap = rag_config.get('chunk_overlap', 200)
        
        self.vector_store = None
        
        # Ensure directory exists
        if not os.path.exists(self.docs_folder):
            os.makedirs(self.docs_folder)

    def ingest_documents(self):
        """
        Loads PDFs, chunks them, and creates a vector store.
        """
        if not os.path.exists(self.docs_folder):
            return f"Error: Folder {self.docs_folder} does not exist."

        # Load PDFs
        loader = DirectoryLoader(self.docs_folder, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        
        if not documents:
            return "No PDF documents found to ingest."

        # Split Text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
        texts = text_splitter.split_documents(documents)

        # Create Vector Store
        embeddings = OpenAIEmbeddings(api_key=self.api_key)
        
        # We use a persistent directory for stability
        self.vector_store = Chroma.from_documents(
            documents=texts, 
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        
        return f"Ingested {len(documents)} documents ({len(texts)} chunks)."

    def query_knowledge_base(self, question, stats_context):
        """
        Queries the RAG system.
        """
        # Lazy Load if needed
        if not self.vector_store:
            try:
                embeddings = OpenAIEmbeddings(api_key=self.api_key)
                self.vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
            except Exception:
                return "Knowledge base not initialized. Please ingest documents first."

        # 1. Retrieve docs (Manual step, replaces Retriever)
        try:
            docs = self.vector_store.similarity_search(question, k=3)
        except Exception:
            return "Error retrieving documents."
            
        if not docs:
            return "No relevant documents found."
            
        # Format citations
        context_text = "\n\n".join([f"[Source: {d.metadata.get('source', 'Unknown')} | Page: {d.metadata.get('page', '?')}]\n{d.page_content}" for d in docs])
        
        # 2. Generate Answer (Manual invoke, replaces RetrievalQA Chain)
        llm = ChatOpenAI(
            model_name=self.config.get('ai_settings', {}).get('model', 'gpt-4o'),
            temperature=0.2, 
            api_key=self.api_key
        )
        
        # We use a simple f-string prompt instead of PromptTemplate
        final_prompt = f"""Use the following pieces of context from regulations and the provided statistical data to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Current Statistical Context:
        {stats_context}

        Regulatory Documents Context:
        {context_text}

        Question: {question}

        Answer (include citations to documents/pages if relevant):"""
                
        # Direct invocation
        response = llm.invoke(final_prompt)
        return response.content