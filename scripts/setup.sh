#!/bin/bash
set -e
echo "Setting up IR project environment..."

python -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
mkdir -p indexes data models

echo ""
echo "Setup complete. Next steps:"
echo "  1. Run: python data/download_msmarco.py"
echo "  2. Run: python scripts/build_index.py"
echo "  3. Run: uvicorn api.main:app --reload"
echo "  4. Open frontend/index.html in your browser"
