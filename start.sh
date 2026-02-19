#!/bin/bash

# Quick Start Script for Resume ATS Tracker
# This script helps you get started quickly with Docker

set -e

echo "Resume ATS Tracker"
echo "===================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo " Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo " Docker Compose is not installed. Please install Docker Compose first."
    echo "   Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

echo " Docker and Docker Compose are installed"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo " Creating .env file from template..."
    cp .env.example .env
    echo "  Please edit .env and add your GOOGLE_API_KEY"
    echo ""
    read -p "Press Enter to continue after you've added your API key, or Ctrl+C to exit..."
fi

# Validate API key is set
source .env
if [ -z "$GOOGLE_API_KEY" ] || [ "$GOOGLE_API_KEY" == "your_google_api_key_here" ]; then
    echo " GOOGLE_API_KEY is not set in .env file"
    echo "   Please edit .env and add your Google API key"
    exit 1
fi

echo "API key is configured"
echo ""

# Build and start services
echo "  Building Docker images..."
docker-compose build

echo ""
echo " Starting services..."
docker-compose up -d

echo ""
echo " Waiting for services to be healthy..."
sleep 5

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    echo " Services are running!"
    echo ""
    echo " Service Status:"
    docker-compose ps
    echo ""
    echo " Access the application:"
    echo "   Web Interface: http://localhost:5000"
    echo "   API Endpoint:  http://localhost:5000/api/analyze"
    echo "   Health Check:  http://localhost:5000/health"
    echo ""
    echo " Useful Commands:"
    echo "   View logs:        docker-compose logs -f"
    echo "   Stop services:    docker-compose down"
    echo "   Restart services: docker-compose restart"
    echo ""
    echo " Documentation: See DOCKER_SETUP.md for detailed information"
else
    echo " Services failed to start. Check logs with: docker-compose logs"
    exit 1
fi
