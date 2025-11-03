from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, LargeBinary, JSON
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from config.database import Base
from datetime import datetime

class Agent(Base):
    __tablename__ = "agents"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    trained_models = relationship("TrainedModel", back_populates="agent", cascade="all, delete-orphan")
    contexts = relationship("ModelContext", back_populates="agent", cascade="all, delete-orphan")

class TrainedModel(Base):
    __tablename__ = "trained_models"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)
    model_type = Column(String(50), nullable=False)  # xgboost, rag, transformer, etc.
    model_name = Column(String(200), nullable=False)
    status = Column(String(20), default="training")  # training, completed, failed
    
    # Model parameters and weights stored as JSON
    parameters = Column(JSON)
    weights_blob = Column(LargeBinary, nullable=True)  # For binary model weights (small models)
    weights_file_path = Column(String(500), nullable=True)  # File path for large model weights
    
    # Training metadata
    training_documents_count = Column(Integer, default=0)
    training_datetime = Column(DateTime, default=datetime.utcnow)
    accuracy = Column(Float, nullable=True)
    
    # Relationships
    agent = relationship("Agent", back_populates="trained_models")
    training_data = relationship("TrainingData", back_populates="model", cascade="all, delete-orphan")

class TrainingData(Base):
    __tablename__ = "training_data"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("trained_models.id"), nullable=False)
    document_type = Column(String(50))  # excel, csv, txt, pdf, image, plain_text
    document_name = Column(String(500))
    content = Column(Text)  # Processed text content
    meta_data = Column(JSON)  # Additional metadata
    embedding = Column(Vector(384))  # Vector embedding using pgvector (384 for all-MiniLM-L6-v2)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    model = relationship("TrainedModel", back_populates="training_data")

class ModelContext(Base):
    __tablename__ = "model_contexts"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)
    context_name = Column(String(200))
    context_data = Column(Text)
    context_embedding = Column(JSON)  # Legacy JSON embedding (kept for compatibility)
    context_embedding_vector = Column(Vector(384))  # Vector embedding using pgvector
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    agent = relationship("Agent", back_populates="contexts")

class QueryHistory(Base):
    __tablename__ = "query_history"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("trained_models.id"), nullable=False)
    query_text = Column(Text, nullable=False)
    response = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    model = relationship("TrainedModel")

