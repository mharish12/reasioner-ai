# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FRONTEND (React)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Dashboard  │  │   Agents     │  │   Training   │     │
│  │   Component  │  │   Component  │  │   Component  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Query      │  │   Database   │  │   Services   │     │
│  │   Component  │  │   View       │  │   (API)      │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────────┬────────────────────────────────┘
                             │ HTTP/REST API
                             │ (Axios)
┌────────────────────────────▼────────────────────────────────┐
│                    BACKEND (FastAPI)                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                 API ENDPOINTS                        │  │
│  │  /api/agents    /api/train   /api/query             │  │
│  │  /api/models    /api/unlearn /api/contexts          │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │               BUSINESS LOGIC LAYER                   │  │
│  │  ┌──────────────┐  ┌──────────────┐                │  │
│  │  │   Trainer    │  │   Processor  │                │  │
│  │  │   Services   │  │   Utilities  │                │  │
│  │  │              │  │              │                │  │
│  │  │ • XGBoost    │  │ • File       │                │  │
│  │  │ • RAG        │  │   Processor  │                │  │
│  │  │ • Transformer│  │ • Validator  │                │  │
│  │  └──────────────┘  └──────────────┘                │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │               DATA ACCESS LAYER                      │  │
│  │  ┌──────────────┐  ┌──────────────┐                │  │
│  │  │   SQLAlchemy │  │   Models     │                │  │
│  │  │   ORM        │  │   & Schemas  │                │  │
│  │  └──────────────┘  └──────────────┘                │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────┘
                             │ SQL
                             │
┌────────────────────────────▼────────────────────────────────┐
│              POSTGRESQL DATABASE                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   agents     │  │ trained_     │  │ training_    │     │
│  │              │  │ models       │  │ data         │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ model_       │  │ query_       │  │              │     │
│  │ contexts     │  │ history      │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### Frontend Architecture

**Technology Stack:**

- React 18.2 for UI
- Vite for build tooling
- Tailwind CSS for styling
- Axios for HTTP requests
- Lucide React for icons

**Component Structure:**

```
src/
├── App.jsx                    # Main app component
├── main.jsx                   # Entry point
├── index.css                  # Global styles
├── components/
│   ├── Dashboard.jsx          # Main dashboard container
│   ├── AgentManagement.jsx    # Agent CRUD operations
│   ├── ModelTraining.jsx      # Model training interface
│   └── ModelQuery.jsx         # Query interface
├── services/
│   └── api.js                 # API service layer
├── components/
│   ├── services.js            # Utility services
│   └── utils.js               # Helper functions
```

**State Management:**

- Component-level state with React hooks
- Props drilling for parent-child communication
- API service layer for server communication

### Backend Architecture

**Technology Stack:**

- FastAPI for REST API
- SQLAlchemy for ORM
- Pydantic for data validation
- PostgreSQL as database

**Layered Architecture:**

```
backend/
├── main.py                    # Application entry point
├── config/
│   └── database.py            # Database configuration
├── models/
│   ├── database_models.py     # SQLAlchemy models
│   └── schemas.py             # Pydantic schemas
├── services/
│   └── model_trainer.py       # Business logic
├── utils/
│   └── file_processor.py      # File utilities
└── api/                       # API routes (reserved)
```

**API Design:**

- RESTful endpoints
- JSON request/response format
- Form data for file uploads
- Swagger/OpenAPI documentation

### Database Architecture

**Entity Relationship:**

```
agents (1) ────< (N) trained_models
agents (1) ────< (N) model_contexts
trained_models (1) ────< (N) training_data
trained_models (1) ────< (N) query_history
```

**Tables:**

1. **agents**: Agent metadata
2. **trained_models**: Model information and weights
3. **training_data**: Document storage
4. **model_contexts**: Agent-specific context
5. **query_history**: Query logs

## Model Training Flow

```
User Uploads Files
        ↓
File Processor (CSV/Excel/TXT)
        ↓
Text Extraction & Chunking
        ↓
Model-Specific Processing
        ↓
┌─────────────────────────┬──────────────┬──────────────┐
│   XGBoost Trainer       │ RAG Trainer  │Transformer   │
│   - TF-IDF Vectorize    │ - Embed      │ - Tokenize   │
│   - Train Classifier    │ - Index      │ - Fine-tune  │
│   - Serialize Model     │ - Store Docs │ - Save State │
└─────────────────────────┴──────────────┴──────────────┘
        ↓
Store Model Weights in PostgreSQL
        ↓
Update Model Status: Completed
```

## Query Flow

```
User Enters Query
        ↓
Select Model
        ↓
Load Model Weights from DB
        ↓
Model-Specific Prediction
        ↓
┌─────────────────────────┬──────────────┬──────────────┐
│   XGBoost               │ RAG          │Transformer   │
│   - Vectorize Query     │ - Embed      │ - Tokenize   │
│   - Predict Class       │ - Search     │ - Generate   │
│   - Return Confidence   │ - Retrieve   │ - Decode     │
└─────────────────────────┴──────────────┴──────────────┘
        ↓
Save to Query History
        ↓
Return Response to User
```

## Data Flow

### Training Data Flow

1. User uploads files or enters text
2. Frontend sends to `/api/train/` endpoint
3. Backend processes files using file_processor
4. Trainer service trains model
5. Model weights serialized and stored in DB
6. Training metadata saved
7. Status updated to "completed"

### Query Data Flow

1. User enters query text
2. Frontend sends to `/api/query/` endpoint
3. Backend loads model from DB
4. Model makes prediction
5. Response saved to query_history
6. Response returned to user

### Unlearn Data Flow

1. User requests data removal
2. Training data deleted from DB
3. If remaining data exists:
   - Model retrained
   - Weights updated
4. If no data remains:
   - Model status set to "unlearned"
   - Weights cleared

## Security Considerations

**Current Implementation:**

- CORS enabled for local development
- SQL injection prevention via SQLAlchemy ORM
- Input validation via Pydantic schemas

**Production Recommendations:**

- Add authentication/authorization (JWT)
- Implement rate limiting
- Add HTTPS/SSL encryption
- Implement API key management
- Add input sanitization
- Enable database backups
- Add logging and monitoring
- Implement audit trails

## Scalability Considerations

**Current:**

- Single instance deployment
- Direct database connections
- In-memory model loading

**Horizontal Scaling Options:**

- Add load balancer
- Use connection pooling
- Implement Redis for caching
- Use message queue for async tasks
- Add container orchestration (K8s)
- Implement CDN for static assets

**Performance Optimizations:**

- Model quantization
- Batch processing
- Async query handling
- Database indexing
- Response caching
- Model versioning

## Deployment Options

### Development

- Local PostgreSQL
- Direct Python execution
- Vite dev server

### Docker

- PostgreSQL container
- Backend container
- Frontend container
- docker-compose orchestration

### Production

- Cloud PostgreSQL (AWS RDS, GCP Cloud SQL)
- Container registry (Docker Hub)
- Orchestration (Kubernetes, ECS)
- Load balancer
- Auto-scaling groups

## Monitoring & Observability

**Recommendations:**

- Application logging (Python logging)
- API monitoring (Prometheus)
- Error tracking (Sentry)
- Performance monitoring (New Relic)
- Database monitoring (pg_stat)
- User analytics
- Model performance metrics

## Future Enhancements

**Short-term:**

- WebSocket support for real-time updates
- Progress tracking for training
- Model comparison tools
- Advanced analytics dashboard

**Long-term:**

- Distributed training support
- Multi-GPU training
- Model versioning system
- AutoML capabilities
- Integration with cloud ML services
- Collaborative features
