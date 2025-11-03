-- Migration: Enable pgvector extension and add vector columns
-- This migration enables pgvector for vector embeddings storage

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Add vector column to training_data table
-- This will store embeddings of dimension 384 (default for all-MiniLM-L6-v2)
-- You can adjust the dimension based on your embedding model
ALTER TABLE training_data 
ADD COLUMN IF NOT EXISTS embedding vector(384);

-- Create index on embedding column for efficient similarity search
CREATE INDEX IF NOT EXISTS training_data_embedding_idx 
ON training_data 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Add vector column to model_contexts table if needed
ALTER TABLE model_contexts 
ADD COLUMN IF NOT EXISTS context_embedding_vector vector(384);

-- Create index on context_embedding_vector
CREATE INDEX IF NOT EXISTS model_contexts_embedding_idx 
ON model_contexts 
USING ivfflat (context_embedding_vector vector_cosine_ops)
WITH (lists = 100);

-- Note: If you're using a different embedding model, adjust the dimension:
-- - all-MiniLM-L6-v2: 384 dimensions
-- - all-mpnet-base-v2: 768 dimensions
-- - sentence-transformers/all-MiniLM-L12-v2: 384 dimensions
-- - OpenAI text-embedding-ada-002: 1536 dimensions

