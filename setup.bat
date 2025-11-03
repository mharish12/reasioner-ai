@echo off
REM AI Platform Setup Script for Windows

echo 🚀 Starting AI Platform Setup...

REM Check if Docker is installed
where docker >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo 📦 Docker detected. Using Docker setup...
    
    REM Start PostgreSQL
    echo 🗄️  Starting PostgreSQL...
    docker-compose up -d postgres
    
    REM Wait for PostgreSQL to be ready
    echo ⏳ Waiting for PostgreSQL to be ready...
    timeout /t 5 /nobreak
    
    REM Start backend
    echo 🔧 Starting backend...
    docker-compose up -d backend
    
    REM Start frontend
    echo 🎨 Starting frontend...
    docker-compose up -d frontend
    
    echo ✅ Setup complete!
    echo.
    echo 🌐 Access the application:
    echo    Frontend: http://localhost:3000
    echo    Backend API: http://localhost:8000
    echo    API Docs: http://localhost:8000/docs
    echo.
    echo 📊 View logs: docker-compose logs -f
    echo 🛑 Stop services: docker-compose down
    
) else (
    echo 🐍 Docker not found. Setting up locally...
    
    REM Backend setup
    echo 🔧 Setting up backend...
    cd backend
    
    if not exist "venv" (
        echo Creating virtual environment...
        python -m venv venv
    )
    
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
    
    echo Installing Python dependencies...
    pip install -r requirements.txt
    
    if not exist ".env" (
        echo Creating .env file...
        copy .env.example .env
    )
    
    echo ✅ Backend setup complete!
    echo.
    echo To start backend:
    echo   cd backend
    echo   venv\Scripts\activate.bat
    echo   python run.py
    
    cd ..
    
    REM Frontend setup
    echo 🎨 Setting up frontend...
    cd frontend
    
    if not exist "node_modules" (
        echo Installing Node dependencies...
        call npm install
    )
    
    echo ✅ Frontend setup complete!
    echo.
    echo To start frontend:
    echo   cd frontend
    echo   npm run dev
    
    cd ..
    
    echo.
    echo ✅ Local setup complete!
    echo.
    echo 📝 Make sure PostgreSQL is running and configured in backend\.env
    echo 🌐 Start backend and frontend in separate terminals
)

echo.
echo 📚 Read README.md for more information

pause

