# AI Model Training Platform

A comprehensive AI platform for training and managing machine learning models with an intuitive React UI. Users can select different model types (XGBoost, RAG, Transformers), train them using documents or plain text, query the trained models, and manage training data.

## Features

### 🤖 Agent Management

- Create and manage multiple AI agents
- Each agent can have multiple trained models
- Contextual isolation of models and data per agent

### 🎓 Model Training

- **XGBoost**: Classification and regression models
- **RAG (Retrieval Augmented Generation)**: Document-based Q&A systems
- **Transformers**: Text generation models
- Support for multiple file formats:
  - **Documents**: Excel (.xlsx, .xls), CSV (.csv), Text files (.txt)
  - **PDF**: PDF files with text extraction (pdfplumber/PyPDF2)
  - **Images**: OCR text extraction from images (JPG, PNG, GIF, BMP, TIFF, WebP)
  - **Plain text input**

### 🔍 Query Interface

- Interactive query interface for trained models
- Real-time responses
- Query history tracking
- Model-specific queries

### 🗑️ Data Management

- Unlearn functionality to remove training data
- Automatic model retraining after data removal
- Persistent storage in PostgreSQL

### 📊 Database Storage

- Model parameters and weights stored in PostgreSQL
- Training data metadata
- Agent-specific contexts
- Query history

## Technology Stack

### Backend

- **FastAPI**: Modern Python web framework
- **SQLAlchemy**: ORM for database management
- **PostgreSQL**: Relational database
- **XGBoost**: Gradient boosting framework
- **Transformers**: Hugging Face transformers library
- **Sentence Transformers**: Embeddings for RAG
- **FAISS**: Vector similarity search
- **LangChain**: LLM orchestration
- **PDF Processing**: PyPDF2, pdfplumber
- **OCR**: Tesseract OCR, Pillow

### Frontend

- **React**: UI library
- **Vite**: Build tool and dev server
- **Tailwind CSS**: Utility-first CSS
- **Axios**: HTTP client
- **Lucide React**: Icon library

### Database

- **PostgreSQL 15**: Primary database

## Project Structure

```
ai/
├── backend/
│   ├── api/              # API routes (if needed)
│   ├── config/           # Configuration files
│   │   └── database.py   # Database connection
│   ├── models/           # Database and Pydantic models
│   │   ├── database_models.py  # SQLAlchemy models
│   │   └── schemas.py          # Pydantic schemas
│   ├── services/         # Business logic
│   │   └── model_trainer.py    # Model training implementations
│   ├── utils/            # Utility functions
│   │   └── file_processor.py   # File processing logic
│   ├── main.py          # FastAPI application
│   ├── requirements.txt # Python dependencies
│   └── Dockerfile       # Backend container config
├── frontend/
│   ├── src/
│   │   ├── components/  # React components
│   │   │   ├── Dashboard.jsx
│   │   │   ├── AgentManagement.jsx
│   │   │   ├── ModelTraining.jsx
│   │   │   └── ModelQuery.jsx
│   │   ├── services/    # API services
│   │   │   └── api.js
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   ├── vite.config.js
│   └── Dockerfile
├── docker-compose.yml    # Multi-container setup
└── README.md
```

## Installation

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL 15+
- Docker and Docker Compose (optional)

**For PDF and Image processing:**

- Tesseract OCR (for image text extraction)
  - macOS: `brew install tesseract`
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
  - Windows: Download from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

### Option 1: Docker Setup (Recommended)

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd ai
   ```

2. **Start services with Docker Compose**

   ```bash
   docker-compose up -d
   ```

3. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Option 2: Local Development Setup

#### Backend Setup

1. **Create and activate virtual environment**

   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   **Note**: For PDF and image processing, ensure Tesseract OCR is installed on your system (see Prerequisites above).

3. **Set up PostgreSQL database**

   ```bash
   # Create database
   createdb ai_platform

   # Or using psql
   psql -U postgres
   CREATE DATABASE ai_platform;
   ```

4. **Configure environment**

   ```bash
   cp .env.example .env
   # Edit .env with your database credentials
   ```

5. **Run database migrations**

   ```bash
   # Tables are created automatically by SQLAlchemy
   python main.py
   ```

6. **Start the backend server**
   ```bash
   python run.py
   # or
   uvicorn main:app --reload
   ```

#### Frontend Setup

1. **Navigate to frontend directory**

   ```bash
   cd frontend
   ```

2. **Install dependencies**

   ```bash
   npm install
   ```

3. **Start development server**

   ```bash
   npm run dev
   ```

4. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000

## Usage

### Creating an Agent

1. Navigate to the **Agents** tab
2. Click **Create Agent**
3. Enter agent name and description
4. Click **Create Agent**

### Training a Model

1. Go to the **Model Training** tab
2. Select an agent (create one if needed)
3. Choose model type:
   - **XGBoost**: For classification/regression tasks
   - **RAG**: For document-based question answering
   - **Transformer**: For text generation
4. Enter a model name
5. Upload files (Excel, CSV, TXT) or enter plain text
6. Click **Start Training**

### Querying a Model

1. Navigate to the **Query Models** tab
2. Select an agent and trained model
3. Enter your question in the text area
4. Click **Send Query**
5. View the response and query history

### Unlearning Data

1. Use the API endpoint `/api/unlearn/`
2. Specify the model ID and document IDs to remove
3. The system will automatically retrain with remaining data

### API Documentation

Interactive API documentation is available at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Agents

- `GET /api/agents/` - List all agents
- `POST /api/agents/` - Create an agent
- `GET /api/agents/{id}` - Get agent details
- `DELETE /api/agents/{id}` - Delete an agent

### Models

- `GET /api/models/` - List all models
- `POST /api/train/` - Train a new model
- `GET /api/models/{id}` - Get model details

### Queries

- `POST /api/query/` - Query a trained model

### Unlearn

- `POST /api/unlearn/` - Remove training data

### Contexts

- `GET /api/contexts/` - List contexts
- `POST /api/contexts/` - Create context

## Database Schema

### Agents

- id, name, description, created_at, updated_at

### Trained Models

- id, agent_id, model_type, model_name, status
- parameters, weights_blob
- training_documents_count, training_datetime, accuracy

### Training Data

- id, model_id, document_type, document_name
- content, metadata, uploaded_at

### Model Contexts

- id, agent_id, context_name, context_data
- context_embedding, created_at, updated_at

### Query History

- id, model_id, query_text, response, timestamp

## Model Details

### XGBoost

- Uses TF-IDF for text feature extraction
- Supports classification and regression
- Configurable hyperparameters

### RAG

- Uses Sentence Transformers for embeddings
- FAISS for vector similarity search
- Retrieves top-k most relevant documents

### Transformers

- Supports Hugging Face models (default: distilgpt2)
- Fine-tuned on provided text
- Configurable training parameters

## Development

### Running Tests

```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

### Code Formatting

```bash
# Backend
black backend/
isort backend/

# Frontend
npm run format
```

## File Processing Details

### Supported Formats

**Documents:**

- Excel files (.xlsx, .xls): Row-by-row text extraction with column metadata
- CSV files (.csv): Row-by-row text extraction with column metadata
- Text files (.txt): Paragraph-based chunking

**PDF Files:**

- Primary: pdfplumber (better text extraction)
- Fallback: PyPDF2
- Page-by-page extraction with metadata
- Automatic text chunking

**Images:**

- Supported formats: JPG, JPEG, PNG, GIF, BMP, TIFF, TIF, WebP
- OCR using Tesseract
- Paragraph-based text extraction
- Image metadata (dimensions, mode)
- Fallback for images with no text

### Processing Behavior

- All files are processed and chunked automatically
- Metadata preserved for each chunk
- Images without text are stored with file information
- PDFs maintain page structure
- Large files are handled efficiently

## Troubleshooting

### Database Connection Issues

- Ensure PostgreSQL is running
- Verify DATABASE_URL in .env file
- Check database credentials

### PDF/Image Processing Issues

- **No PDF libraries**: Install `pip install PyPDF2 pdfplumber`
- **Tesseract not found**: Install Tesseract OCR on your system
- **OCR fails**: Ensure image quality is good for text recognition
- **Large files**: Processing may take time depending on file size

### Port Already in Use

- Change ports in docker-compose.yml or configuration files
- Kill processes using the ports

### Model Training Fails

- Check file formats are supported
- Verify sufficient memory is available
- Review backend logs for errors

### Frontend Not Connecting to Backend

- Verify API_BASE_URL in frontend
- Check CORS settings in backend
- Ensure backend is running on correct port

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License

## Support

For issues and questions:

- Open an issue on GitHub
- Check the documentation
- Review API docs at /docs endpoint

## Future Enhancements

- [ ] Support for more model types (BERT, GPT, etc.)
- [ ] Real-time training progress updates
- [ ] Model versioning and comparisons
- [ ] Export/import trained models
- [ ] Advanced analytics and visualizations
- [ ] Authentication and user management
- [ ] Multi-user collaboration
- [ ] Cloud storage integration
- [ ] Model performance monitoring
- [ ] A/B testing for models
