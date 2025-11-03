"""
LangChain-based RAG trainer with pgvector support
This trainer uses LangChain for retrieval-augmented generation to reduce hallucinations
and provide accurate responses based on the training data.
"""
import os
import json
from typing import List, Dict, Any, Tuple, Union, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
import numpy as np

from models.database_models import TrainedModel, TrainingData

# LangChain imports
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import PGVector
    from langchain.chains import RetrievalQA
    from langchain.schema import BaseRetriever, Document
    from langchain.prompts import PromptTemplate
except ImportError:
    # Fallback for older versions
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores.pgvector import PGVector
    from langchain.chains import RetrievalQA
    from langchain.schema import BaseRetriever, Document
    from langchain.prompts import PromptTemplate

# LLM imports - will be imported dynamically in _get_llm method


class LangChainRAGTrainer:
    """LangChain RAG trainer using pgvector for embeddings storage"""
    
    def __init__(self, db: Session):
        self.db = db
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        self.connection_string = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ai_platform")
        
    def _get_embedding_model(self, model_name: str = "all-MiniLM-L6-v2"):
        """Get embedding model using HuggingFace"""
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def _get_llm(self, llm_type: str = "ollama", llm_config: Optional[Dict] = None):
        """
        Get LLM instance based on type.
        Supports: ollama (free), openai (paid), huggingface (free)
        """
        config = llm_config or {}
        
        if llm_type.lower() == "ollama":
            try:
                from langchain_ollama import OllamaLLM
                model_name = config.get("model_name", "llama2")
                base_url = config.get("base_url", "http://localhost:11434")
                
                return OllamaLLM(
                    model=model_name,
                    base_url=base_url,
                    temperature=config.get("temperature", 0.7),
                )
            except ImportError:
                raise ImportError("langchain-ollama not installed. Install with: pip install langchain-ollama")
        
        elif llm_type.lower() == "openai":
            try:
                from langchain_openai import ChatOpenAI
                api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or provide in config.")
                
                return ChatOpenAI(
                    model=config.get("model_name", "gpt-3.5-turbo"),
                    temperature=config.get("temperature", 0.7),
                    openai_api_key=api_key
                )
            except ImportError:
                raise ImportError("langchain-openai not installed. Install with: pip install langchain-openai")
        
        elif llm_type.lower() == "huggingface":
            # Use HuggingFace models (free but slower)
            try:
                from langchain_community.llms import HuggingFacePipeline
                from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
                
                model_name = config.get("model_name", "microsoft/DialoGPT-medium")
                
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name)
                
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=512,
                    temperature=config.get("temperature", 0.7),
                )
                
                return HuggingFacePipeline(pipeline=pipe)
            except ImportError as e:
                raise ImportError(f"HuggingFace LLM requires langchain-community and transformers: {e}")
        
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}. Supported: ollama, openai, huggingface")
    
    def train(
        self, 
        documents: List[str], 
        model_id: int,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Train RAG model by storing embeddings in pgvector
        Returns: (model_id as string, metadata)
        """
        params = parameters or {}
        embedding_model_name = params.get("embedding_model", "all-MiniLM-L6-v2")
        
        # Get embedding model
        embeddings = self._get_embedding_model(embedding_model_name)
        
        # Update embedding dimension based on model
        if "mpnet" in embedding_model_name.lower():
            self.embedding_dim = 768
        elif "ada" in embedding_model_name.lower() or "openai" in embedding_model_name.lower():
            self.embedding_dim = 1536
        else:
            self.embedding_dim = 384
        
        # Generate embeddings for all documents
        print(f"Generating embeddings for {len(documents)} documents...")
        embedding_list = embeddings.embed_documents(documents)
        
        # Store embeddings in database
        training_data_records = self.db.query(TrainingData).filter(
            TrainingData.model_id == model_id
        ).all()
        
        if len(training_data_records) != len(documents):
            raise ValueError(f"Mismatch: {len(training_data_records)} training data records but {len(documents)} documents")
        
        # Update each training data record with its embedding
        for i, record in enumerate(training_data_records):
            # Convert embedding to numpy array and then to list for pgvector
            embedding_array = np.array(embedding_list[i])
            record.embedding = embedding_array.tolist()
        
        self.db.commit()
        
        metadata = {
            "num_documents": len(documents),
            "embedding_dim": self.embedding_dim,
            "embedding_model": embedding_model_name
        }
        
        # Return model_id as string (we don't need to store model weights, everything is in DB)
        return str(model_id), metadata
    
    def predict(
        self, 
        model_id: int,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Make prediction using LangChain RAG chain
        This retrieves relevant documents and generates response using LLM
        """
        params = parameters or {}
        
        # Get model from database
        model = self.db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # Get embedding model
        embedding_model_name = params.get("embedding_model", "all-MiniLM-L6-v2")
        if model.parameters and "embedding_model" in model.parameters:
            embedding_model_name = model.parameters.get("embedding_model", embedding_model_name)
        
        embeddings = self._get_embedding_model(embedding_model_name)
        
        # Create pgvector connection string for LangChain
        # LangChain PGVector expects a connection string with collection name
        collection_name = f"training_data_model_{model_id}"
        
        # Create a custom retriever that queries our training_data table
        retriever = self._create_custom_retriever(model_id, embeddings, params)
        
        # Get LLM
        llm_type = params.get("llm_type", "ollama")
        llm_config = params.get("llm_config", {})
        
        try:
            llm = self._get_llm(llm_type, llm_config)
        except Exception as e:
            # Fallback to a simple retrieval if LLM fails
            return self._simple_retrieval(query, retriever)
        
        # Create RAG prompt template
        prompt_template = """Use the following pieces of context to answer the question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
This helps prevent hallucinations and ensures accurate responses.

Context: {context}

Question: {question}

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval QA chain
        try:
            # Try newer LangChain API first
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
            # Execute query
            result = qa_chain({"query": query})
            response = result.get("result", "No response generated")
            
            # Include source information if available
            source_docs = result.get("source_documents", [])
            if source_docs:
                response += f"\n\n[Based on {len(source_docs)} relevant document(s)]"
            
            return response
        except Exception as chain_error:
            # Try with invoke method for newer LangChain versions
            try:
                from langchain.chains.question_answering import load_qa_chain
                qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)
                
                # Get relevant documents
                docs = retriever.get_relevant_documents(query)
                result = qa_chain.invoke({"input_documents": docs, "question": query})
                
                if isinstance(result, dict):
                    response = result.get("output_text", "No response generated")
                else:
                    response = str(result)
                
                response += f"\n\n[Based on {len(docs)} relevant document(s)]"
                return response
            except Exception as e:
                # Fallback to simple retrieval if both methods fail
                import traceback
                print(f"LangChain chain error: {chain_error}\nFallback error: {e}")
                return self._simple_retrieval(query, retriever)
    
    def _create_custom_retriever(
        self, 
        model_id: int, 
        embeddings: HuggingFaceEmbeddings,
        params: Dict[str, Any]
    ) -> BaseRetriever:
        """Create a custom retriever that uses pgvector similarity search"""
        
        db = self.db  # Capture db reference
        k = params.get("top_k", 3)
        
        class CustomPGRetriever(BaseRetriever):
            def __init__(self, model_id: int, db_session: Session, embeddings_model: HuggingFaceEmbeddings, k: int = 3):
                super().__init__()
                self.model_id = model_id
                self.db_session = db_session
                self.embeddings_model = embeddings_model
                self.k = k
            
            def _get_relevant_documents(self, query: str) -> List[Document]:
                # Generate query embedding
                query_embedding = self.embeddings_model.embed_query(query)
                query_embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
                
                # Use pgvector cosine similarity search
                sql_query = text("""
                    SELECT id, content, meta_data, 
                           1 - (embedding <=> :query_embedding::vector) as similarity
                    FROM training_data
                    WHERE model_id = :model_id
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> :query_embedding::vector
                    LIMIT :k
                """)
                
                results = self.db_session.execute(
                    sql_query,
                    {
                        "query_embedding": query_embedding_str,
                        "model_id": self.model_id,
                        "k": self.k
                    }
                ).fetchall()
                
                documents = []
                for row in results:
                    # Handle meta_data - could be dict or string
                    meta = row.meta_data if row.meta_data else {}
                    if isinstance(row.meta_data, str):
                        try:
                            meta = json.loads(row.meta_data)
                        except:
                            meta = {}
                    
                    doc = Document(
                        page_content=row.content or "",
                        metadata={
                            "id": row.id,
                            "similarity": float(row.similarity),
                            **meta
                        }
                    )
                    documents.append(doc)
                
                return documents
        
        return CustomPGRetriever(model_id, db, embeddings, k)
    
    def _simple_retrieval(self, query: str, retriever: BaseRetriever) -> str:
        """Simple retrieval fallback without LLM"""
        docs = retriever.get_relevant_documents(query)
        
        if not docs:
            return "No relevant documents found for your query."
        
        response = "Based on the following relevant information:\n\n"
        for i, doc in enumerate(docs[:3], 1):
            response += f"{i}. {doc.page_content[:200]}...\n"
        
        return response

