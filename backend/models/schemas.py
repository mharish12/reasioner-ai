from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Dict, Any

# Agent Schemas
class AgentBase(BaseModel):
    name: str
    description: Optional[str] = None

class AgentCreate(AgentBase):
    pass

class Agent(AgentBase):
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# Trained Model Schemas
class TrainedModelBase(BaseModel):
    agent_id: int
    model_type: str
    model_name: str
    parameters: Optional[Dict[str, Any]] = None

class TrainedModelCreate(TrainedModelBase):
    pass

class TrainedModelUpdate(BaseModel):
    status: Optional[str] = None
    accuracy: Optional[float] = None

class TrainedModel(TrainedModelBase):
    id: int
    status: str
    training_documents_count: int
    training_datetime: datetime
    accuracy: Optional[float] = None
    
    class Config:
        from_attributes = True

# Training Data Schemas
class TrainingDataBase(BaseModel):
    model_id: int
    document_type: str
    document_name: str
    content: str
    meta_data: Optional[Dict[str, Any]] = None

class TrainingDataCreate(TrainingDataBase):
    pass

class TrainingData(TrainingDataBase):
    id: int
    uploaded_at: datetime
    
    class Config:
        from_attributes = True

# Model Context Schemas
class ModelContextBase(BaseModel):
    agent_id: int
    context_name: str
    context_data: str

class ModelContextCreate(ModelContextBase):
    pass

class ModelContext(ModelContextBase):
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# Query Schemas
class QueryRequest(BaseModel):
    model_id: int
    query_text: str

class QueryResponse(BaseModel):
    model_id: int
    response: str
    timestamp: datetime

# File Upload Schema
class FileUploadResponse(BaseModel):
    model_id: int
    document_count: int
    message: str

# Training Request Schema
class TrainingRequest(BaseModel):
    agent_id: int
    model_type: str
    model_name: str
    parameters: Optional[Dict[str, Any]] = None
    document_ids: Optional[List[int]] = None

# Unlearn Request Schema
class UnlearnRequest(BaseModel):
    model_id: int
    document_ids: Optional[List[int]] = None  # If None, unlearn all

