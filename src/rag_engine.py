import os
import yaml
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

class RagEngine:
    """
    RAG Engine for "Chat with your Data & Docs".
    """
    def __init__(self, config_path="config.yaml"):
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.docs_folder = self.config['rag']['documents_folder']
        self.chunk_size = self.config['rag']['chunk_size']
        self.chunk_overlap = self.config['rag']['chunk_overlap']
        self.vector_store = None
        self.qa_chain = None

    def ingest_documents(self):
        """
        Loads PDFs, chunks them, and creates a vector store.
        """
        if not os.path.exists(self.docs_folder):
            os.makedirs(self.docs_folder)
            return "Documents folder created. Please add PDFs."

        # Load PDFs
        loader = DirectoryLoader(self.docs_folder, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()

        if not documents:
            return "No PDF documents found to ingest."

        # Split Text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        texts = text_splitter.split_documents(documents)

        # Create Vector Store
        embeddings = OpenAIEmbeddings(api_key=self.api_key)
        self.vector_store = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=None # In-memory for now as per requirements "simple"
        )

        # Initialize QA Chain
        llm = ChatOpenAI(
            model_name=self.config['ai_settings']['model'],
            temperature=0,
            api_key=self.api_key
        )

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        # Custom Prompt to include Stats Context
        template = """Use the following pieces of context from regulations and the provided statistical data to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Current Statistical Context:
        {stats_context}

        Regulatory Documents Context:
        {context}

        Question: {question}

        Answer (include citations to documents/pages if relevant):"""

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question", "stats_context"]
        )

        # We use a custom chain setup to inject stats_context dynamically?
        # Standard RetrievalQA doesn't easily allow extra inputs.
        # We can implement query_knowledge_base to do retrieval then LLM call manually
        # OR use a Chain that allows extra vars.
        # Let's do manual retrieval + generation for flexibility with 'stats_context'.

        return f"Ingested {len(documents)} documents ({len(texts)} chunks)."

    def query_knowledge_base(self, question, stats_context):
        """
        Queries the RAG system.
        """
        if not self.vector_store:
            return "Knowledge base not initialized. Please ingest documents first."

        # Retrieve docs
        docs = self.vector_store.similarity_search(question, k=3)
        context_text = "\n\n".join([f"[Source: {d.metadata.get('source', 'Unknown')} Page: {d.metadata.get('page', '?')}]\n{d.page_content}" for d in docs])

        # Generate Answer
        llm = ChatOpenAI(
            model_name=self.config['ai_settings']['model'],
            temperature=0.2,
            api_key=self.api_key
        )

        final_prompt = self.prompt.format(
            context=context_text,
            question=question,
            stats_context=stats_context
        )

        response = llm.invoke(final_prompt)
        return response.content
