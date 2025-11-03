#!/bin/bash

# AI Platform Setup Script

set -e

echo "🚀 Starting AI Platform Setup..."

# Check if Docker is installed
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
    echo "📦 Docker detected. Using Docker setup..."
    
    # Start PostgreSQL
    echo "🗄️  Starting PostgreSQL..."
    docker-compose up -d postgres
    
    # Wait for PostgreSQL to be ready
    echo "⏳ Waiting for PostgreSQL to be ready..."
    sleep 5
    
    # Start backend
    echo "🔧 Starting backend..."
    docker-compose up -d backend
    
    # Start frontend
    echo "🎨 Starting frontend..."
    docker-compose up -d frontend
    
    echo "✅ Setup complete!"
    echo ""
    echo "🌐 Access the application:"
    echo "   Frontend: http://localhost:3000"
    echo "   Backend API: http://localhost:8000"
    echo "   API Docs: http://localhost:8000/docs"
    echo ""
    echo "📊 View logs: docker-compose logs -f"
    echo "🛑 Stop services: docker-compose down"
    
else
    echo "🐍 Docker not found. Setting up locally..."
    
    # Backend setup
    echo "🔧 Setting up backend..."
    cd backend
    
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    echo "Activating virtual environment..."
    source venv/bin/activate || source venv/Scripts/activate
    
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
    
    if [ ! -f ".env" ]; then
        echo "Creating .env file..."
        cp .env.example .env
    fi
    
    echo "✅ Backend setup complete!"
    echo ""
    echo "To start backend:"
    echo "  cd backend"
    echo "  source venv/bin/activate  # or venv\\Scripts\\activate on Windows"
    echo "  python run.py"
    
    cd ..
    
    # Frontend setup
    echo "🎨 Setting up frontend..."
    cd frontend
    
    if [ ! -d "node_modules" ]; then
        echo "Installing Node dependencies..."
        npm install
    fi
    
    echo "✅ Frontend setup complete!"
    echo ""
    echo "To start frontend:"
    echo "  cd frontend"
    echo "  npm run dev"
    
    cd ..
    
    echo ""
    echo "✅ Local setup complete!"
    echo ""
    echo "📝 Make sure PostgreSQL is running and configured in backend/.env"
    echo "🌐 Start backend and frontend in separate terminals"
fi

echo ""
echo "📚 Read README.md for more information"

