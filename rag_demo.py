import os
import time
import streamlit as st
from typing import List, Dict, Any
import PyPDF2
import fitz  # PyMuPDF
from io import BytesIO

# Vector DB and embeddings - Fixed imports
import chromadb
from chromadb.config import Settings
import requests
import numpy as np
from sentence_transformers import SentenceTransformer

# LLM integration
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter

class SimpleEmbeddings:
    """Simple embedding wrapper to avoid LangChain dependency issues"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        embedding = self.model.encode([text], convert_to_tensor=False)
        return embedding[0].tolist()

class PDFParser:
    """Enhanced PDF parser with multiple extraction methods"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_pypdf2(self, pdf_file) -> str:
        """Extract text using PyPDF2"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"PyPDF2 extraction failed: {e}")
            return ""
    
    def extract_text_pymupdf(self, pdf_file) -> str:
        """Extract text using PyMuPDF (better for complex PDFs)"""
        try:
            pdf_file.seek(0)  # Reset file pointer
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            text = ""
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text += page.get_text() + "\n"
            doc.close()
            return text
        except Exception as e:
            st.error(f"PyMuPDF extraction failed: {e}")
            return ""
    
    def parse_pdf(self, pdf_file) -> List[str]:
        """Parse PDF and return text chunks"""
        # Try PyMuPDF first (better quality), fallback to PyPDF2
        text = self.extract_text_pymupdf(pdf_file)
        if not text.strip():
            pdf_file.seek(0)  # Reset for second attempt
            text = self.extract_text_pypdf2(pdf_file)
        
        if not text.strip():
            raise ValueError("Could not extract text from PDF")
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        return chunks

class VectorDBManager:
    """Chroma vector database manager with fixed imports"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embedding_model = SimpleEmbeddings()
        
        # Ensure directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize Chroma client with fixed settings
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = None
        self.documents = []  # Store documents for retrieval
        self.embeddings = []  # Store embeddings
        self.metadata = []    # Store metadata
    
    def create_collection(self, collection_name: str = "rag_documents"):
        """Create or get collection"""
        try:
            # Get or create collection with metadata
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            return True
        except Exception as e:
            st.error(f"Failed to create collection: {e}")
            return False
    
    def add_documents(self, texts: List[str], metadatas: List[Dict] = None):
        """Add documents to vector store"""
        try:
            if metadatas is None:
                metadatas = [{"source": f"doc_{i}"} for i in range(len(texts))]
            
            # Generate embeddings
            embeddings = self.embedding_model.embed_documents(texts)
            
            # Generate IDs
            ids = [f"doc_{len(self.documents) + i}" for i in range(len(texts))]
            
            # Add to Chroma collection
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            # Store locally for retrieval
            self.documents.extend(texts)
            self.embeddings.extend(embeddings)
            self.metadata.extend(metadatas)
            
            return True
        except Exception as e:
            st.error(f"Failed to add documents: {e}")
            return False
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Perform similarity search"""
        try:
            if not self.collection:
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_model.embed_query(query)
            
            # Search in Chroma
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, len(self.documents))
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        "content": doc,
                        "metadata": results['metadatas'][0][i] if results['metadatas'][0] else {},
                        "score": 1 - results['distances'][0][i]  # Convert distance to similarity
                    })
            
            return formatted_results
        except Exception as e:
            st.error(f"Search failed: {e}")
            return []

class LLMManager:
    """Ollama LLM manager with error handling"""
    
    def __init__(self, model_name: str = "llama3.1:8b"):
        self.model_name = model_name
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Ollama client with error handling"""
        try:
            self.client = ollama.Client()
            self._check_model_availability()
        except Exception as e:
            st.error(f"Failed to initialize Ollama client: {e}")
            st.error("Please ensure Ollama is installed and running")
    
    def _check_model_availability(self):
        """Check if model is available locally"""
        try:
            if not self.client:
                return
                
            models = self.client.list()
            available_models = [m['name'] for m in models['models']]
            
            if self.model_name not in available_models:
                st.warning(f"""
                Model {self.model_name} not found. 
                
                Please install it using:
                ```
                ollama pull {self.model_name}
                ```
                
                Available models: {', '.join(available_models) if available_models else 'None'}
                """)
        except Exception as e:
            st.error(f"Failed to check model availability: {e}")
    
    def generate_response(self, prompt: str, context: str = "", max_tokens: int = 512) -> str:
        """Generate response with context"""
        if not self.client:
            return "Error: Ollama client not initialized. Please check your Ollama installation."
        
        try:
            full_prompt = f"""Context: {context}

Question: {prompt}

Please provide a comprehensive answer based on the context above. If the context doesn't contain relevant information, please state that clearly.

Answer:"""
            
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': full_prompt
                    }
                ],
                options={
                    'temperature': 0.7,
                    'num_predict': max_tokens,
                    'top_k': 40,
                    'top_p': 0.9,
                }
            )
            
            return response['message']['content']
        except Exception as e:
            error_msg = str(e)
            if "connection" in error_msg.lower():
                return "Error: Cannot connect to Ollama. Please ensure Ollama is running."
            elif "model" in error_msg.lower():
                return f"Error: Model {self.model_name} not available. Please pull the model first."
            else:
                return f"Error generating response: {error_msg}"

class RAGPipeline:
    """Complete RAG pipeline orchestrator"""
    
    def __init__(self):
        self.pdf_parser = PDFParser()
        self.vector_db = VectorDBManager()
        self.llm = LLMManager()
        self.is_initialized = False
    
    def initialize(self):
        """Initialize all components"""
        if self.vector_db.create_collection():
            self.is_initialized = True
            return True
        return False
    
    def process_documents(self, uploaded_files) -> bool:
        """Process uploaded PDF documents"""
        try:
            all_chunks = []
            all_metadata = []
            
            for file_idx, uploaded_file in enumerate(uploaded_files):
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    chunks = self.pdf_parser.parse_pdf(uploaded_file)
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        all_chunks.append(chunk)
                        all_metadata.append({
                            "source": uploaded_file.name,
                            "chunk_id": chunk_idx,
                            "file_id": file_idx
                        })
            
            # Add to vector database
            success = self.vector_db.add_documents(all_chunks, all_metadata)
            if success:
                st.success(f"Successfully processed {len(all_chunks)} chunks from {len(uploaded_files)} documents")
            return success
            
        except Exception as e:
            st.error(f"Document processing failed: {e}")
            return False
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system"""
        start_time = time.time()
        
        # Retrieve relevant documents
        relevant_docs = self.vector_db.similarity_search(question, k=5)
        retrieval_time = time.time() - start_time
        
        # Prepare context
        context = "\n\n".join([doc["content"] for doc in relevant_docs])
        
        # Generate response
        generation_start = time.time()
        response = self.llm.generate_response(question, context)
        generation_time = time.time() - generation_start
        
        total_time = time.time() - start_time
        
        return {
            "answer": response,
            "sources": relevant_docs,
            "metrics": {
                "retrieval_time": round(retrieval_time, 3),
                "generation_time": round(generation_time, 3),
                "total_time": round(total_time, 3),
                "num_sources": len(relevant_docs)
            }
        }

def main():
    st.set_page_config(
        page_title="RAG Demo - High Performance Q&A System",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 High-Performance RAG Demo")
    st.markdown("**Optimized for Speed & Accuracy** | Local LLM + Vector Search")
    
    # System status check
    with st.sidebar:
        st.header("🔍 System Status")
        
        # Check Ollama connection
        try:
            client = ollama.Client()
            models = client.list()
            st.success("✅ Ollama Connected")
            st.info(f"Available models: {len(models['models'])}")
        except Exception as e:
            st.error("❌ Ollama Not Connected")
            st.error("Please start Ollama service")
    
    # Initialize session state
    if 'rag_pipeline' not in st.session_state:
        with st.spinner("Initializing RAG pipeline..."):
            st.session_state.rag_pipeline = RAGPipeline()
            if not st.session_state.rag_pipeline.initialize():
                st.error("Failed to initialize RAG pipeline")
                st.stop()
        st.success("RAG pipeline initialized successfully!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Model Information")
        st.info("""
        **Vector DB**: Chroma (Local)
        **Embeddings**: all-MiniLM-L6-v2
        **LLM**: Llama 3.1 8B (Ollama)
        **Parser**: PyMuPDF + PyPDF2
        """)
        
        st.subheader("Performance Metrics")
        if 'last_metrics' in st.session_state:
            metrics = st.session_state.last_metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Retrieval", f"{metrics['retrieval_time']}s")
                st.metric("Sources", metrics['num_sources'])
            with col2:
                st.metric("Generation", f"{metrics['generation_time']}s")
                st.metric("Total", f"{metrics['total_time']}s")
    
    # Main interface
    tab1, tab2 = st.tabs(["📄 Document Upload", "💬 Q&A Interface"])
    
    with tab1:
        st.header("Document Upload & Processing")
        
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF documents to create your knowledge base"
        )
        
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                success = st.session_state.rag_pipeline.process_documents(uploaded_files)
                if success:
                    st.balloons()
    
    with tab2:
        st.header("Ask Questions")
        
        # Quick example questions
        st.subheader("💡 Try these example questions:")
        example_questions = [
            "What are the main topics covered in the documents?",
            "Can you summarize the key findings?",
            "What are the recommendations mentioned?",
            "Explain the methodology used in the research."
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(example_questions):
            with cols[i % 2]:
                if st.button(question, key=f"example_{i}"):
                    st.session_state.current_question = question
        
        # Question input
        question = st.text_input(
            "Your Question:",
            value=st.session_state.get('current_question', ''),
            placeholder="Ask anything about your uploaded documents..."
        )
        
        if st.button("Get Answer", type="primary") and question:
            with st.spinner("🔍 Searching documents and generating answer..."):
                result = st.session_state.rag_pipeline.query(question)
                
                # Store metrics for sidebar
                st.session_state.last_metrics = result['metrics']
                
                # Display answer
                st.subheader("💡 Answer")
                st.write(result['answer'])
                
                # Display sources
                with st.expander("📚 Sources & Context", expanded=False):
                    for i, source in enumerate(result['sources']):
                        st.markdown(f"**Source {i+1}** (Score: {source['score']:.3f})")
                        st.markdown(f"*File: {source['metadata']['source']}*")
                        st.text_area(
                            f"Content {i+1}:",
                            source['content'][:500] + "..." if len(source['content']) > 500 else source['content'],
                            height=100,
                            key=f"source_{i}"
                        )
                        st.divider()
                
                # Performance metrics
                with st.expander("⚡ Performance Metrics", expanded=True):
                    metrics = result['metrics']
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("🔍 Retrieval Time", f"{metrics['retrieval_time']}s")
                    with col2:
                        st.metric("🤖 Generation Time", f"{metrics['generation_time']}s")
                    with col3:
                        st.metric("⏱️ Total Time", f"{metrics['total_time']}s")
                    with col4:
                        st.metric("📄 Sources Used", metrics['num_sources'])

if __name__ == "__main__":
    main()