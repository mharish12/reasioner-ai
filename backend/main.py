from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import warnings

# Suppress multiprocessing warnings
warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing')

from config.database import engine, get_db, Base
from models import database_models, schemas
from services.model_trainer import get_trainer
from utils.file_processor import process_file, process_plain_text

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Model Training Platform", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== AGENT ENDPOINTS ====================

@app.post("/api/agents/", response_model=schemas.Agent)
async def create_agent(agent: schemas.AgentCreate, db: Session = Depends(get_db)):
    """Create a new agent"""
    try:
        db_agent = database_models.Agent(**agent.model_dump())
        db.add(db_agent)
        db.commit()
        db.refresh(db_agent)
        return db_agent
    except Exception as e:
        db.rollback()
        error_msg = str(e)
        if "unique constraint" in error_msg.lower() or "already exists" in error_msg.lower():
            raise HTTPException(status_code=400, detail=f"Agent name '{agent.name}' already exists")
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {error_msg}")

@app.get("/api/agents/", response_model=List[schemas.Agent])
async def get_agents(db: Session = Depends(get_db)):
    """Get all agents"""
    return db.query(database_models.Agent).all()

@app.get("/api/agents/{agent_id}", response_model=schemas.Agent)
async def get_agent(agent_id: int, db: Session = Depends(get_db)):
    """Get a specific agent"""
    agent = db.query(database_models.Agent).filter(database_models.Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@app.delete("/api/agents/{agent_id}")
async def delete_agent(agent_id: int, db: Session = Depends(get_db)):
    """Delete an agent and all its models"""
    agent = db.query(database_models.Agent).filter(database_models.Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    try:
        db.delete(agent)
        db.commit()
        return {"message": "Agent deleted successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete agent: {str(e)}")

# ==================== TRAINING ENDPOINTS ====================

@app.post("/api/train/", response_model=schemas.TrainedModel)
async def train_model(
    agent_id: int = Form(...),
    model_type: str = Form(...),
    model_name: str = Form(...),
    parameters: Optional[str] = Form(None),  # JSON string
    files: List[UploadFile] = File(None),
    plain_text: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Train a model with uploaded files or plain text"""
    import json
    
    # Validate inputs
    if not model_type or not model_name:
        raise HTTPException(status_code=400, detail="Model type and name are required")
    
    if model_type not in ['xgboost', 'rag', 'langchain_rag', 'transformer']:
        raise HTTPException(status_code=400, detail="Invalid model type. Must be: xgboost, rag, langchain_rag, or transformer")
    
    # Parse parameters
    model_params = None
    if parameters:
        try:
            model_params = json.loads(parameters)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in parameters: {str(e)}")
    
    # Create model record
    try:
        db_model = database_models.TrainedModel(
            agent_id=agent_id,
            model_type=model_type,
            model_name=model_name,
            parameters=model_params,
            status="training"
        )
        db.add(db_model)
        db.commit()
        db.refresh(db_model)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create model record: {str(e)}")
    
    try:
        # Process documents
        all_documents = []
        
        # Process uploaded files
        if files:
            for file in files:
                try:
                    content = await file.read()
                    documents = process_file(content, file.filename)
                    # Get file type from processor
                    from utils.file_processor import get_file_type
                    file_type = get_file_type(file.filename)
                    
                    for text, metadata in documents:
                        db_training_data = database_models.TrainingData(
                            model_id=db_model.id,
                            document_type=file_type,
                            document_name=file.filename,
                            content=text,
                            meta_data=metadata
                        )
                        db.add(db_training_data)
                        all_documents.append(text)
                except Exception as file_error:
                    # If file processing fails, rollback and return error
                    db_model.status = "failed"
                    db.commit()
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Failed to process file '{file.filename}': {str(file_error)}"
                    )
        
        # Process plain text
        if plain_text:
            documents = process_plain_text(plain_text)
            for text, metadata in documents:
                db_training_data = database_models.TrainingData(
                    model_id=db_model.id,
                    document_type='plain_text',
                    document_name='user_input',
                    content=text,
                    meta_data=metadata
                )
                db.add(db_training_data)
                all_documents.append(text)
        
        if not all_documents:
            db_model.status = "failed"
            db.commit()
            raise HTTPException(status_code=400, detail="No documents provided for training")
        
        # Train the model
        trainer = get_trainer(model_type, db)
        
        # Handle langchain_rag trainer which has different interface
        if model_type == 'langchain_rag':
            from services.langchain_rag_trainer import LangChainRAGTrainer
            if isinstance(trainer, LangChainRAGTrainer):
                # LangChainRAGTrainer needs model_id and stores embeddings in DB
                model_data, metadata = trainer.train(all_documents, db_model.id, model_params)
            else:
                model_data, metadata = trainer.train(all_documents, model_params)
        else:
            model_data, metadata = trainer.train(all_documents, model_params)
        
        # Update model record - handle both file paths and bytes
        db_model.status = "completed"
        db_model.training_documents_count = len(all_documents)
        db_model.accuracy = metadata.get('accuracy')
        db_model.parameters = {**(model_params or {}), **metadata}
        
        # Store model data appropriately based on size
        # langchain_rag stores everything in DB, so no weights needed
        if model_type == 'langchain_rag':
            db_model.weights_blob = None
            db_model.weights_file_path = None
        elif isinstance(model_data, str):
            # File path for large models (transformers)
            db_model.weights_file_path = model_data
            db_model.weights_blob = None
        else:
            # Bytes for small models (XGBoost, RAG)
            db_model.weights_blob = model_data
            db_model.weights_file_path = None
        
        try:
            db.commit()
            db.refresh(db_model)
        except Exception as commit_error:
            # If commit fails, clean up file if it was created
            if isinstance(model_data, str) and os.path.exists(model_data):
                try:
                    os.remove(model_data)
                except:
                    pass
            db.rollback()
            raise commit_error
        
        return db_model
        
    except HTTPException:
        # Re-raise HTTPExceptions as-is
        raise
    except Exception as e:
        # Handle any other exceptions
        error_msg = str(e)
        # Provide more helpful error messages
        if "accelerate" in error_msg.lower():
            error_msg = "Transformers training requires 'accelerate' package. Install with: pip install accelerate>=0.26.0"
        if "memory" in error_msg.lower() or "alloc" in error_msg.lower():
            error_msg = "Model too large for database. File storage should handle this automatically."
        
        # Try to mark model as failed
        try:
            db_model.status = "failed"
            db.commit()
        except:
            db.rollback()
        
        raise HTTPException(status_code=500, detail=f"Training failed: {error_msg}")

@app.get("/api/models/", response_model=List[schemas.TrainedModel])
async def get_models(agent_id: Optional[int] = None, db: Session = Depends(get_db)):
    """Get all trained models, optionally filtered by agent"""
    query = db.query(database_models.TrainedModel)
    if agent_id:
        query = query.filter(database_models.TrainedModel.agent_id == agent_id)
    return query.all()

@app.get("/api/models/{model_id}", response_model=schemas.TrainedModel)
async def get_model(model_id: int, db: Session = Depends(get_db)):
    """Get a specific trained model"""
    model = db.query(database_models.TrainedModel).filter(database_models.TrainedModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model

# ==================== QUERY ENDPOINTS ====================

@app.post("/api/query/", response_model=schemas.QueryResponse)
async def query_model(request: schemas.QueryRequest, db: Session = Depends(get_db)):
    """Query a trained model"""
    from datetime import datetime
    
    # Get model
    model = db.query(database_models.TrainedModel).filter(
        database_models.TrainedModel.id == request.model_id
    ).first()
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if model.status != "completed":
        raise HTTPException(status_code=400, detail="Model is not trained yet")
    
    try:
        # Validate query input
        if not request.query_text or not request.query_text.strip():
            raise HTTPException(status_code=400, detail="Query text cannot be empty")
        
        # Get trainer
        trainer = get_trainer(model.model_type, db)
        
        # Get model data - handle both file path and blob storage
        # langchain_rag doesn't need model_data, everything is in DB
        model_data = None
        if model.model_type != 'langchain_rag':
            if model.weights_file_path:
                model_data = model.weights_file_path
            elif model.weights_blob:
                model_data = model.weights_blob
            else:
                raise HTTPException(status_code=404, detail="Model weights not found")
        
        # Make prediction - handle langchain_rag differently
        if model.model_type == 'langchain_rag':
            from services.langchain_rag_trainer import LangChainRAGTrainer
            if isinstance(trainer, LangChainRAGTrainer):
                # LangChainRAGTrainer doesn't need model_data, uses DB
                response = trainer.predict(model.id, request.query_text, model.parameters)
            else:
                response = trainer.predict(model_data, request.query_text)
        else:
            response = trainer.predict(model_data, request.query_text)
        
        # Save query history
        try:
            db_query = database_models.QueryHistory(
                model_id=model.id,
                query_text=request.query_text,
                response=response
            )
            db.add(db_query)
            db.commit()
        except Exception as db_error:
            # If history save fails, log but don't fail the query
            print(f"Warning: Failed to save query history: {db_error}")
            db.rollback()
        
        return schemas.QueryResponse(
            model_id=model.id,
            response=response,
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        # Re-raise HTTPExceptions as-is
        raise
    except Exception as e:
        error_msg = str(e)
        # Provide more helpful error messages
        if "model not found" in error_msg.lower():
            raise HTTPException(status_code=404, detail="Trained model data not found")
        if "weights" in error_msg.lower():
            raise HTTPException(status_code=500, detail="Model weights appear to be corrupted or missing")
        raise HTTPException(status_code=500, detail=f"Query failed: {error_msg}")

# ==================== UNLEARN ENDPOINTS ====================

@app.post("/api/unlearn/")
async def unlearn_data(request: schemas.UnlearnRequest, db: Session = Depends(get_db)):
    """Unlearn training data from a model"""
    
    # Get model
    model = db.query(database_models.TrainedModel).filter(
        database_models.TrainedModel.id == request.model_id
    ).first()
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Delete training data
    if request.document_ids:
        training_data = db.query(database_models.TrainingData).filter(
            database_models.TrainingData.model_id == model.id,
            database_models.TrainingData.id.in_(request.document_ids)
        ).all()
    else:
        # Delete all training data
        training_data = db.query(database_models.TrainingData).filter(
            database_models.TrainingData.model_id == model.id
        ).all()
    
    # Check if any training data was found
    if not training_data:
        if request.document_ids:
            raise HTTPException(status_code=404, detail="No training data found with specified IDs")
        else:
            raise HTTPException(status_code=404, detail="No training data found for this model")
    
    # Delete training data
    for data in training_data:
        db.delete(data)
    
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete training data: {str(e)}")
    
    # Retrain the model if there's remaining data
    remaining_data = db.query(database_models.TrainingData).filter(
        database_models.TrainingData.model_id == model.id
    ).all()
    
    if remaining_data:
        try:
            # Collect remaining documents
            documents = [data.content for data in remaining_data]
            
            # Retrain
            trainer = get_trainer(model.model_type, db)
            
            # Handle langchain_rag trainer which has different interface
            if model.model_type == 'langchain_rag':
                from services.langchain_rag_trainer import LangChainRAGTrainer
                if isinstance(trainer, LangChainRAGTrainer):
                    model_data, metadata = trainer.train(documents, model.id, model.parameters)
                else:
                    model_data, metadata = trainer.train(documents, model.parameters)
            else:
                model_data, metadata = trainer.train(documents, model.parameters)
            
            # Update model - handle both file paths and bytes
            # langchain_rag stores everything in DB
            if model.model_type == 'langchain_rag':
                model.weights_blob = None
                model.weights_file_path = None
            else:
                # First, clean up old file if exists
                if model.weights_file_path and os.path.exists(model.weights_file_path):
                    try:
                        os.remove(model.weights_file_path)
                    except:
                        pass
                
                # Store new model data
                if isinstance(model_data, str):
                    model.weights_file_path = model_data
                    model.weights_blob = None
                else:
                    model.weights_blob = model_data
                    model.weights_file_path = None
            
            model.training_documents_count = len(documents)
            model.parameters = {**(model.parameters or {}), **metadata}
            
            db.commit()
            
            return {"message": "Model retrained with remaining data", "remaining_documents": len(documents)}
        except Exception as e:
            error_msg = str(e)
            # Provide helpful error message for retraining failures
            return {
                "message": f"Data removed but could not retrain model: {error_msg}", 
                "remaining_documents": len(documents),
                "warning": "Model may not function properly without retraining"
            }
    
    else:
        # No data left, mark model as unlearned
        # Clean up file if exists
        if model.weights_file_path and os.path.exists(model.weights_file_path):
            try:
                os.remove(model.weights_file_path)
            except:
                pass
        
        model.status = "unlearned"
        model.weights_blob = None
        model.weights_file_path = None
        db.commit()
        
        return {"message": "All training data removed. Model is unlearned."}

# ==================== CONTEXT ENDPOINTS ====================

@app.post("/api/contexts/", response_model=schemas.ModelContext)
async def create_context(context: schemas.ModelContextCreate, db: Session = Depends(get_db)):
    """Create context for an agent"""
    db_context = database_models.ModelContext(**context.model_dump())
    db.add(db_context)
    db.commit()
    db.refresh(db_context)
    return db_context

@app.get("/api/contexts/", response_model=List[schemas.ModelContext])
async def get_contexts(agent_id: Optional[int] = None, db: Session = Depends(get_db)):
    """Get contexts, optionally filtered by agent"""
    query = db.query(database_models.ModelContext)
    if agent_id:
        query = query.filter(database_models.ModelContext.agent_id == agent_id)
    return query.all()

# ==================== HEALTH CHECK ====================

@app.get("/")
async def root():
    return {"message": "AI Model Training Platform API", "status": "running"}

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

