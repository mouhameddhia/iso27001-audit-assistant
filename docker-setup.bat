@echo off
setlocal enabledelayedexpansion

echo.
echo 🚀 ISO 27001 Audit Assistant - Docker Setup
echo ==========================================
echo.

:: Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

echo ✅ Docker found

:: Build and start services
echo.
echo 📦 Building and starting services...
echo.

docker compose up -d --build

echo.
echo ⏳ Waiting for services to be healthy...
timeout /t 10

:: Check if all services are running
echo.
echo 📊 Service Status:
docker compose ps

echo.
echo 🔧 Pulling LLM models (this may take several minutes)...
echo    Pulling llama3:8b (this is large ~4GB)...
docker exec iso27001_ollama ollama pull llama3:8b

echo.
echo    Pulling bge-m3 (embeddings model)...
docker exec iso27001_ollama ollama pull bge-m3

echo.
echo ✅ Setup complete!
echo.
echo 🌐 Access the application:
echo    - Frontend:     http://localhost:5173
echo    - Backend API:  http://localhost:8000
echo    - API Docs:     http://localhost:8000/docs
echo    - Qdrant UI:    http://localhost:6333/dashboard
echo.
echo 📝 Admin credentials (check backend logs for first login):
echo    - Username: admin
echo    - Password: (shown in backend logs on first start)
echo.
echo 🛑 To stop all services:
echo    docker compose down
echo.
echo 📜 To view logs:
echo    docker compose logs -f
echo.
pause
