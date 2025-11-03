# Quick Start Guide

Get up and running with the AI Platform in 5 minutes!

## Prerequisites

- Docker & Docker Compose installed
- OR Python 3.11+ and Node.js 18+ with PostgreSQL

## Option 1: Docker (Easiest)

```bash
# 1. Clone the repository
git clone <repository-url>
cd ai

# 2. Start all services
docker-compose up -d

# 3. Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

That's it! The platform is ready to use.

## Option 2: Local Setup

### Step 1: Database Setup

Install and start PostgreSQL:

```bash
# macOS (using Homebrew)
brew install postgresql
brew services start postgresql

# Linux (Ubuntu/Debian)
sudo apt-get install postgresql postgresql-contrib
sudo systemctl start postgresql

# Windows
# Download and install from https://www.postgresql.org/download/windows/
```

Create the database:

```bash
psql -U postgres
CREATE DATABASE ai_platform;
\q
```

### Step 2: Backend Setup

```bash
# Navigate to backend
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your PostgreSQL credentials

# Start backend
python run.py
```

Backend should be running on http://localhost:8000

### Step 3: Frontend Setup

Open a new terminal:

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start frontend
npm run dev
```

Frontend should be running on http://localhost:3000

## First Steps

1. **Create an Agent**

   - Open http://localhost:3000
   - Go to "Agents" tab
   - Click "Create Agent"
   - Enter name and description
   - Click "Create Agent"

2. **Train a Model**

   - Go to "Model Training" tab
   - Select your agent
   - Choose model type (RAG recommended for beginners)
   - Enter model name (e.g., "My First Model")
   - Upload files (CSV, TXT, Excel, PDF, Images) or enter plain text
   - Click "Start Training"
   - Wait for training to complete

3. **Query the Model**
   - Go to "Query Models" tab
   - Select your agent and trained model
   - Enter a question in the text area
   - Click "Send Query"
   - View the response!

## Example Training Data

### Plain Text Example

```
Artificial intelligence is transforming the world.
Machine learning enables computers to learn from data.
Deep learning uses neural networks for complex tasks.
Natural language processing understands human language.
Computer vision allows machines to interpret images.
```

### CSV Example

Create a file `data.csv`:

```csv
topic,description
AI,Artificial intelligence
ML,Machine learning
DL,Deep learning
NLP,Natural language processing
CV,Computer vision
```

Upload this file in the Model Training interface.

### PDF Example

You can upload any PDF document. The system will:

- Extract text from each page
- Maintain page structure
- Store metadata (page numbers, total pages)

### Image Example

Upload an image with text (JPG, PNG, etc.). The system will:

- Extract text using OCR (Tesseract)
- Preserve image metadata
- Handle images without text gracefully

**Note**: For image OCR, ensure Tesseract is installed on your system.

## Testing the API

You can test the API directly using curl or the Swagger UI:

### Swagger UI

Visit http://localhost:8000/docs for interactive API documentation.

### Using curl

```bash
# Create an agent
curl -X POST http://localhost:8000/api/agents/ \
  -H "Content-Type: application/json" \
  -d '{"name": "Test Agent", "description": "Testing"}'

# List agents
curl http://localhost:8000/api/agents/

# Query a model (replace MODEL_ID with actual ID)
curl -X POST http://localhost:8000/api/query/ \
  -H "Content-Type: application/json" \
  -d '{"model_id": 1, "query_text": "What is artificial intelligence?"}'
```

## Troubleshooting

### Database Connection Error

- Ensure PostgreSQL is running
- Check credentials in `backend/.env`
- Verify database `ai_platform` exists

### Port Already in Use

```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9  # macOS/Linux
taskkill /F /PID <PID>  # Windows

# Kill process on port 3000
lsof -ti:3000 | xargs kill -9  # macOS/Linux
taskkill /F /PID <PID>  # Windows
```

### Module Not Found Errors

```bash
# Backend
cd backend
source venv/bin/activate
pip install -r requirements.txt

# Frontend
cd frontend
rm -rf node_modules
npm install
```

### Docker Issues

```bash
# Restart all services
docker-compose down
docker-compose up -d

# View logs
docker-compose logs -f

# Rebuild containers
docker-compose up -d --build
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore different model types (XGBoost, RAG, Transformers)
- Try the unlearn functionality
- Check out the Database view to see stored data
- Review API documentation at /docs endpoint

## Getting Help

- Check the [README.md](README.md) for detailed documentation
- Review API docs at http://localhost:8000/docs
- Open an issue on GitHub
- Check backend logs for errors

## Tips

1. **Start with RAG**: Easiest to understand and works well with documents
2. **Use small files first**: Test with small datasets before large ones
3. **Check model status**: Ensure model shows "completed" before querying
4. **Explore API docs**: Use Swagger UI to understand available endpoints
5. **Monitor logs**: Keep an eye on backend logs for debugging

Happy training! 🚀
