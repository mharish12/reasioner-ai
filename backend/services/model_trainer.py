import pickle
import json
import os
import time
from typing import List, Dict, Any, Tuple, Union
from sqlalchemy.orm import Session
from models.database_models import TrainedModel, TrainingData
from utils.file_processor import process_file

class BaseModelTrainer:
    """Base class for all model trainers"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def train(self, documents: List[str], parameters: Dict[str, Any] = None) -> Tuple[Union[bytes, str], Any]:
        """Train the model and return model data (bytes or file path) and metadata"""
        raise NotImplementedError
    
    def predict(self, model_data: Union[bytes, str], query: str) -> str:
        """Make prediction on query - model_data can be bytes or file path"""
        raise NotImplementedError
    
    def unlearn(self, model_id: int, document_ids: List[int] = None) -> bool:
        """Remove training data and retrain if necessary"""
        raise NotImplementedError

class XGBoostTrainer(BaseModelTrainer):
    """XGBoost model trainer for classification/regression"""
    
    def train(self, documents: List[str], parameters: Dict[str, Any] = None) -> Tuple[Any, Any]:
        try:
            import xgboost as xgb
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.model_selection import train_test_split
            
            # Default parameters
            params = parameters or {}
            xgb_params = params.get('xgb_params', {
                'objective': 'multi:softprob',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100
            })
            
            # Feature extraction using TF-IDF
            vectorizer = TfidfVectorizer(max_features=100)
            X = vectorizer.fit_transform(documents)
            
            # For demo purposes, create dummy labels
            # In real scenario, labels should come from documents
            import numpy as np
            y = np.random.randint(0, 3, size=len(documents))
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            # Train model
            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X_train, y_train)
            
            # Serialize
            model_bytes = pickle.dumps({
                'model': model,
                'vectorizer': vectorizer
            })
            
            metadata = {
                'accuracy': float(model.score(X_test, y_test)),
                'features': len(vectorizer.vocabulary_)
            }
            
            return model_bytes, metadata
            
        except Exception as e:
            raise Exception(f"XGBoost training error: {str(e)}")
    
    def predict(self, model_data: Union[bytes, str], query: str) -> str:
        try:
            import xgboost as xgb
            
            # Load model - handle both file path and bytes
            if isinstance(model_data, str):
                with open(model_data, 'rb') as f:
                    model_dict = pickle.load(f)
            else:
                model_dict = pickle.loads(model_data)
            
            model = model_dict['model']
            vectorizer = model_dict['vectorizer']
            
            # Predict
            X = vectorizer.transform([query])
            prediction = model.predict(X)
            probabilities = model.predict_proba(X)
            
            return f"Prediction: {prediction[0]}, Confidence: {max(probabilities[0]):.2f}"
            
        except Exception as e:
            raise Exception(f"XGBoost prediction error: {str(e)}")

class RAGTrainer(BaseModelTrainer):
    """RAG (Retrieval Augmented Generation) model trainer using FAISS (legacy)"""
    
    def train(self, documents: List[str], parameters: Dict[str, Any] = None) -> Tuple[Any, Any]:
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            import numpy as np
            
            # Initialize embedding model
            params = parameters or {}
            model_name = params.get('embedding_model', 'all-MiniLM-L6-v2')
            embedding_model = SentenceTransformer(model_name)
            
            # Create embeddings
            embeddings = embedding_model.encode(documents, show_progress_bar=True)
            embeddings = np.array(embeddings).astype('float32')
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            
            # Serialize
            model_bytes = pickle.dumps({
                'embedding_model': embedding_model,
                'index': index,
                'documents': documents
            })
            
            metadata = {
                'num_documents': len(documents),
                'embedding_dim': dimension,
                'model_name': model_name
            }
            
            return model_bytes, metadata
            
        except Exception as e:
            raise Exception(f"RAG training error: {str(e)}")
    
    def predict(self, model_data: Union[bytes, str], query: str) -> str:
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            import numpy as np
            
            # Load model - handle both file path and bytes
            if isinstance(model_data, str):
                with open(model_data, 'rb') as f:
                    model_dict = pickle.load(f)
            else:
                model_dict = pickle.loads(model_data)
            
            embedding_model = model_dict['embedding_model']
            index = model_dict['index']
            documents = model_dict['documents']
            
            # Query embedding
            query_embedding = embedding_model.encode([query])
            query_embedding = np.array(query_embedding).astype('float32')
            
            # Search
            k = 3  # Top 3 results
            distances, indices = index.search(query_embedding, k)
            
            # Construct response
            results = []
            for i, idx in enumerate(indices[0]):
                results.append({
                    'document': documents[idx],
                    'distance': float(distances[0][i])
                })
            
            return f"Top {k} relevant documents found:\n" + "\n".join([f"{i+1}. {r['document'][:100]}..." for i, r in enumerate(results)])
            
        except Exception as e:
            raise Exception(f"RAG prediction error: {str(e)}")

class TransformerTrainer(BaseModelTrainer):
    """Transformer model trainer for text generation"""
    
    def train(self, documents: List[str], parameters: Dict[str, Any] = None) -> Tuple[Any, Any]:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
            from datasets import Dataset
            import torch
            
            params = parameters or {}
            model_name = params.get('model_name', 'distilgpt2')
            
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Tokenize documents
            def tokenize_function(examples):
                return tokenizer(examples['text'], truncation=True, padding=True, max_length=512)
            
            dataset = Dataset.from_dict({'text': documents})
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir='./training_output',
                num_train_epochs=params.get('epochs', 1),
                per_device_train_batch_size=params.get('batch_size', 4),
                save_steps=100,
                logging_steps=10,
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
            
            # Train
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
            )
            
            trainer.train()
            
            # Save to file instead of pickle (large models exceed DB size limits)
            os.makedirs('model_storage', exist_ok=True)
            timestamp = int(time.time() * 1000)
            model_file_path = f'model_storage/model_{model_name}_{timestamp}.pkl'
            
            with open(model_file_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'tokenizer': tokenizer
                }, f)
            
            metadata = {
                'model_name': model_name,
                'num_documents': len(documents),
                'epochs': params.get('epochs', 1)
            }
            
            return model_file_path, metadata
            
        except Exception as e:
            raise Exception(f"Transformer training error: {str(e)}")
    
    def predict(self, model_data: Union[bytes, str], query: str) -> str:
        try:
            import torch
            
            # Load model - handle both file path and bytes
            if isinstance(model_data, str):
                # Load from file
                with open(model_data, 'rb') as f:
                    model_dict = pickle.load(f)
            else:
                # Load from bytes
                model_dict = pickle.loads(model_data)
            
            model = model_dict['model']
            tokenizer = model_dict['tokenizer']
            
            # Generate - ensure inputs are on same device as model
            device = next(model.parameters()).device
            inputs = tokenizer.encode(query, return_tensors='pt').to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 50,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
            
        except Exception as e:
            raise Exception(f"Transformer prediction error: {str(e)}")

def get_trainer(model_type: str, db: Session) -> BaseModelTrainer:
    """Factory function to get appropriate trainer"""
    from services.langchain_rag_trainer import LangChainRAGTrainer
    
    trainers = {
        'xgboost': XGBoostTrainer,
        'rag': RAGTrainer,  # Legacy FAISS-based RAG
        'langchain_rag': LangChainRAGTrainer,  # New pgvector + LangChain RAG
        'transformer': TransformerTrainer
    }
    
    trainer_class = trainers.get(model_type.lower())
    if not trainer_class:
        raise Exception(f"Unsupported model type: {model_type}. Supported: xgboost, rag, langchain_rag, transformer")
    
    return trainer_class(db)

