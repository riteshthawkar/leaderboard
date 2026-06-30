#!/bin/bash
# Run the Combined Leaderboard Application

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Combined Vision Leaderboard${NC}"
echo "================================"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${BLUE}Creating .env from .env.example...${NC}"
    cp .env.example .env
    echo -e "${GREEN}✓ .env created. Please update paths in .env file.${NC}"
fi

# Check if venv exists
if [ ! -d venv ]; then
    echo -e "${BLUE}Creating virtual environment...${NC}"
    "$PYTHON_BIN" -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate venv
source venv/bin/activate

# Install/update dependencies
echo -e "${BLUE}Installing dependencies...${NC}"
pip install -r requirements.txt -q
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Run the Flask app
echo -e "${BLUE}Starting Combined Leaderboard Server...${NC}"
export FLASK_PORT="${FLASK_PORT:-5050}"
echo -e "${GREEN}Server running at http://localhost:${FLASK_PORT}${NC}"
echo ""
python backend/web/app.py
