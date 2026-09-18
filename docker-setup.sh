#!/bin/bash

echo "🚀 ISO 27001 Audit Assistant - Docker Setup"
echo "==========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker Desktop first."
    exit 1
fi

# Check if Docker Compose is available
if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not available. Please install Docker Desktop."
    exit 1
fi

echo "✅ Docker and Docker Compose found"
echo ""

# Build and start services
echo "📦 Building and starting services..."
echo ""

docker compose up -d --build

echo ""
echo "⏳ Waiting for services to be healthy..."
sleep 10

# Check if all services are running
echo ""
echo "📊 Service Status:"
docker compose ps

echo ""
echo "🔧 Pulling LLM models (this may take a few minutes)..."
docker exec iso27001_ollama ollama pull llama3:8b
docker exec iso27001_ollama ollama pull bge-m3

echo ""
echo "✅ Setup complete!"
echo ""
echo "🌐 Access the application:"
echo "   - Frontend:     http://localhost:5173"
echo "   - Backend API:  http://localhost:8000"
echo "   - API Docs:     http://localhost:8000/docs"
echo "   - Qdrant UI:    http://localhost:6333/dashboard"
echo ""
echo "📝 Admin credentials (check backend logs for first login):"
echo "   - Username: admin"
echo "   - Password: <shown in backend logs on first start>"
echo ""
echo "🛑 To stop all services:"
echo "   docker compose down"
echo ""
echo "📜 To view logs:"
echo "   docker compose logs -f"
echo ""
