"""
LangChain-based RAG trainer with pgvector support
This trainer uses LangChain for retrieval-augmented generation to reduce hallucinations
and provide accurate responses based on the training data.
"""
import os
import json
import logging
from typing import List, Dict, Any, Tuple, Union, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
import numpy as np
from datetime import datetime

from models.database_models import TrainedModel, TrainingData

logger = logging.getLogger(__name__)

# LangChain imports - Updated for LangChain 0.3.x
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import PGVector
    from langchain.chains import RetrievalQA
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from langchain_core.prompts import PromptTemplate
except ImportError:
    # Fallback for older versions
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain.chains import RetrievalQA
        from langchain.schema import BaseRetriever, Document
        from langchain.prompts import PromptTemplate
    except ImportError:
        # Even older versions
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.chains import RetrievalQA
        from langchain.schema import BaseRetriever, Document
        from langchain.prompts import PromptTemplate

# LLM imports - will be imported dynamically in _get_llm method


# Custom PG Retriever class definition
class CustomPGRetriever(BaseRetriever):
    """Custom retriever that uses pgvector similarity search"""
    
    # Use private attributes to avoid Pydantic validation issues
    # These will be set via object.__setattr__ in __init__
    _model_id: int = None
    _db_session: Session = None
    _embeddings_model: Any = None
    _k: int = 3
    
    def __init__(self, model_id: int, db_session: Session, embeddings_model, k: int = 3, **kwargs):
        # Call super().__init__() first
        super().__init__(**kwargs)
        # Store attributes using object.__setattr__ to bypass Pydantic validation
        # Using private attributes (with underscore) and object.__setattr__
        object.__setattr__(self, '_model_id', model_id)
        object.__setattr__(self, '_db_session', db_session)
        object.__setattr__(self, '_embeddings_model', embeddings_model)
        object.__setattr__(self, '_k', k)
    
    @property
    def model_id(self):
        """Get model_id"""
        return self._model_id
    
    @property
    def db_session(self):
        """Get db_session"""
        return self._db_session
    
    @property
    def embeddings_model(self):
        """Get embeddings_model"""
        return self._embeddings_model
    
    @property
    def k(self):
        """Get k"""
        return self._k
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents using pgvector similarity search"""
        # Generate query embedding
        query_embedding = self.embeddings_model.embed_query(query)
        query_embedding_list = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else list(query_embedding)
        
        logger.debug(f"[LangChain RAG] Query embedding generated (dim: {len(query_embedding_list)})")
        
        # Convert embedding to PostgreSQL array format
        # Use ARRAY constructor to avoid SQLAlchemy parameter binding issues with ::vector
        array_values = ",".join(map(str, query_embedding_list))
        array_str = f"ARRAY[{array_values}]"
        
        # Use pgvector cosine similarity search
        # SQLAlchemy's text() doesn't support :: casting in parameters
        # We embed the ARRAY constructor directly in the SQL string (safe - it's numeric data)
        # Use parameterized queries only for model_id and k
        sql_query = text(f"""
            SELECT id, content, meta_data, 
                   1 - (embedding <=> {array_str}::vector) as similarity
            FROM training_data
            WHERE model_id = :model_id
              AND embedding IS NOT NULL
            ORDER BY embedding <=> {array_str}::vector
            LIMIT :k
        """)
        
        try:
            logger.debug(f"[LangChain RAG] Executing pgvector similarity search...")
            results = self.db_session.execute(
                sql_query,
                {
                    "model_id": self.model_id,
                    "k": self.k
                }
            ).fetchall()
            logger.debug(f"[LangChain RAG] Retrieved {len(results)} documents from pgvector")
        except Exception as e:
            logger.error(f"[LangChain RAG] SQL query failed: {str(e)}", exc_info=True)
            # Fallback: try with string array format
            try:
                query_embedding_str = "[" + ",".join(map(str, query_embedding_list)) + "]"
                query_embedding_escaped = query_embedding_str.replace("'", "''")
                sql_query_alt = text(f"""
                    SELECT id, content, meta_data, 
                           1 - (embedding <=> '{query_embedding_escaped}'::vector) as similarity
                    FROM training_data
                    WHERE model_id = :model_id
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> '{query_embedding_escaped}'::vector
                    LIMIT :k
                """)
                results = self.db_session.execute(
                    sql_query_alt,
                    {
                        "model_id": self.model_id,
                        "k": self.k
                    }
                ).fetchall()
                logger.info(f"[LangChain RAG] Fallback query succeeded, retrieved {len(results)} documents")
            except Exception as e2:
                logger.error(f"[LangChain RAG] Fallback query also failed: {str(e2)}", exc_info=True)
                raise e  # Raise original error
        
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
            except Exception as e:
                raise Exception(f"Failed to load HuggingFace model: {e}. Make sure transformers is installed correctly.")
        
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
        logger.info(f"[LangChain RAG] Starting training for model_id: {model_id}")
        logger.info(f"[LangChain RAG] Number of documents: {len(documents)}")
        
        params = parameters or {}
        embedding_model_name = params.get("embedding_model", "all-MiniLM-L6-v2")
        logger.info(f"[LangChain RAG] Using embedding model: {embedding_model_name}")
        
        # Get embedding model
        logger.info(f"[LangChain RAG] Loading embedding model...")
        start_time = datetime.now()
        embeddings = self._get_embedding_model(embedding_model_name)
        load_duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"[LangChain RAG] Embedding model loaded in {load_duration:.2f} seconds")
        
        # Update embedding dimension based on model
        if "mpnet" in embedding_model_name.lower():
            self.embedding_dim = 768
        elif "ada" in embedding_model_name.lower() or "openai" in embedding_model_name.lower():
            self.embedding_dim = 1536
        else:
            self.embedding_dim = 384
        
        logger.info(f"[LangChain RAG] Embedding dimension: {self.embedding_dim}")
        
        # Generate embeddings for all documents
        logger.info(f"[LangChain RAG] Generating embeddings for {len(documents)} documents...")
        start_time = datetime.now()
        embedding_list = embeddings.embed_documents(documents)
        embedding_duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"[LangChain RAG] Embeddings generated in {embedding_duration:.2f} seconds ({len(documents) / embedding_duration:.2f} docs/sec)")
        
        # Store embeddings in database
        logger.info(f"[LangChain RAG] Retrieving training data records from database...")
        
        # Refresh the session to ensure we see committed records
        self.db.expire_all()
        
        training_data_records = self.db.query(TrainingData).filter(
            TrainingData.model_id == model_id
        ).all()
        
        logger.info(f"[LangChain RAG] Found {len(training_data_records)} training data records")
        
        if len(training_data_records) != len(documents):
            error_msg = f"Mismatch: {len(training_data_records)} training data records but {len(documents)} documents"
            logger.error(f"[LangChain RAG] {error_msg}")
            logger.error(f"[LangChain RAG] Model ID: {model_id}, Documents count: {len(documents)}")
            
            # Try to refresh and query again
            self.db.commit()  # Ensure any pending changes are committed
            self.db.expire_all()  # Clear session cache
            
            training_data_records = self.db.query(TrainingData).filter(
                TrainingData.model_id == model_id
            ).all()
            logger.info(f"[LangChain RAG] After refresh: Found {len(training_data_records)} training data records")
            
            if len(training_data_records) != len(documents):
                raise ValueError(error_msg)
        
        # Update each training data record with its embedding
        logger.info(f"[LangChain RAG] Storing embeddings in database (pgvector)...")
        start_time = datetime.now()
        for i, record in enumerate(training_data_records):
            if (i + 1) % 10 == 0 or i == 0:
                logger.debug(f"[LangChain RAG] Storing embedding {i+1}/{len(training_data_records)}")
            # Convert embedding to numpy array and then to list for pgvector
            embedding_array = np.array(embedding_list[i])
            record.embedding = embedding_array.tolist()
        
        logger.info(f"[LangChain RAG] Committing embeddings to database...")
        self.db.commit()
        store_duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"[LangChain RAG] Embeddings stored in database in {store_duration:.2f} seconds")
        
        metadata = {
            "num_documents": len(documents),
            "embedding_dim": self.embedding_dim,
            "embedding_model": embedding_model_name
        }
        
        logger.info(f"[LangChain RAG] Training completed successfully. Metadata: {json.dumps(metadata, indent=2)}")
        
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
        logger.info(f"[LangChain RAG] Starting prediction for model_id: {model_id}")
        logger.info(f"[LangChain RAG] Query: {query[:100]}...")
        
        params = parameters or {}
        
        # Get model from database
        logger.debug(f"[LangChain RAG] Retrieving model from database...")
        model = self.db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
        if not model:
            logger.error(f"[LangChain RAG] Model {model_id} not found")
            raise ValueError(f"Model {model_id} not found")
        
        logger.info(f"[LangChain RAG] Model found: {model.model_name} (type: {model.model_type})")
        
        # Get embedding model
        embedding_model_name = params.get("embedding_model", "all-MiniLM-L6-v2")
        if model.parameters and "embedding_model" in model.parameters:
            embedding_model_name = model.parameters.get("embedding_model", embedding_model_name)
        
        logger.info(f"[LangChain RAG] Using embedding model: {embedding_model_name}")
        embeddings = self._get_embedding_model(embedding_model_name)
        
        # Create pgvector connection string for LangChain
        # LangChain PGVector expects a connection string with collection name
        collection_name = f"training_data_model_{model_id}"
        
        # Create a custom retriever that queries our training_data table
        retriever = self._create_custom_retriever(model_id, embeddings, params)
        
        # Get LLM
        llm_type = params.get("llm_type", "ollama")
        llm_config = params.get("llm_config", {})
        
        logger.info(f"[LangChain RAG] Using LLM type: {llm_type}")
        logger.debug(f"[LangChain RAG] LLM config: {json.dumps(llm_config, indent=2)}")
        
        try:
            logger.info(f"[LangChain RAG] Loading LLM...")
            start_time = datetime.now()
            llm = self._get_llm(llm_type, llm_config)
            llm_load_duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"[LangChain RAG] LLM loaded in {llm_load_duration:.2f} seconds")
        except Exception as e:
            logger.error(f"[LangChain RAG] Failed to load LLM: {str(e)}", exc_info=True)
            logger.warning(f"[LangChain RAG] Falling back to simple retrieval")
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
            logger.info(f"[LangChain RAG] Creating RAG chain...")
            # Try newer LangChain 0.3.x API with invoke
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
            logger.info(f"[LangChain RAG] Executing RAG query...")
            start_time = datetime.now()
            # Use invoke instead of __call__ for LangChain 0.3.x
            result = qa_chain.invoke({"query": query})
            query_duration = (datetime.now() - start_time).total_seconds()
            
            response = result.get("result", "No response generated")
            
            # Include source information if available
            source_docs = result.get("source_documents", [])
            logger.info(f"[LangChain RAG] Query completed in {query_duration:.2f} seconds")
            logger.info(f"[LangChain RAG] Retrieved {len(source_docs)} source document(s)")
            
            if source_docs:
                response += f"\n\n[Based on {len(source_docs)} relevant document(s)]"
            
            logger.info(f"[LangChain RAG] Response generated (length: {len(response)} chars)")
            return response
        except Exception as chain_error:
            logger.warning(f"[LangChain RAG] Primary chain method failed: {str(chain_error)}")
            # Try alternative approach using load_qa_chain
            try:
                logger.info(f"[LangChain RAG] Trying alternative chain method...")
                from langchain.chains.question_answering import load_qa_chain
                qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)
                
                # Get relevant documents using invoke for LangChain 0.3.x
                docs = retriever.invoke(query) if hasattr(retriever, 'invoke') else retriever.get_relevant_documents(query)
                result = qa_chain.invoke({"input_documents": docs, "question": query})
                
                if isinstance(result, dict):
                    response = result.get("output_text", "No response generated")
                else:
                    response = str(result)
                
                logger.info(f"[LangChain RAG] Alternative method succeeded")
                response += f"\n\n[Based on {len(docs)} relevant document(s)]"
                return response
            except Exception as e:
                # Fallback to simple retrieval if both methods fail
                logger.error(f"[LangChain RAG] All chain methods failed: {str(e)}", exc_info=True)
                logger.warning(f"[LangChain RAG] Falling back to simple retrieval")
                return self._simple_retrieval(query, retriever)
    
    def _create_custom_retriever(
        self, 
        model_id: int, 
        embeddings: HuggingFaceEmbeddings,
        params: Dict[str, Any]
    ) -> BaseRetriever:
        """Create a custom retriever that uses pgvector similarity search"""
        
        k = params.get("top_k", 3)
        logger.debug(f"[LangChain RAG] Creating CustomPGRetriever with model_id={model_id}, k={k}")
        
        # Use the module-level CustomPGRetriever class
        return CustomPGRetriever(model_id, self.db, embeddings, k)
    
    def _simple_retrieval(self, query: str, retriever: BaseRetriever) -> str:
        """Simple retrieval fallback without LLM"""
        # Use invoke for LangChain 0.3.x, fallback to get_relevant_documents for older versions
        try:
            docs = retriever.invoke(query)
        except (AttributeError, TypeError):
            docs = retriever.get_relevant_documents(query)
        
        if not docs:
            return "No relevant documents found for your query."
        
        response = "Based on the following relevant information:\n\n"
        for i, doc in enumerate(docs[:3], 1):
            response += f"{i}. {doc.page_content[:200]}...\n"
        
        return response

