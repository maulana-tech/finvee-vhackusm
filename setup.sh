#!/bin/bash
# ============================================================
# AEGIS SME — One-Command Setup Script (Mac/Linux)
# Team Finvee | Varsity Hackathon 2026
# Usage: bash setup.sh
# ============================================================

set -e

echo ""
echo "⚔️  AEGIS SME — Setup Script"
echo "   Team Finvee | Varsity Hackathon 2026"
echo "============================================"

# Step 1: Install UV if not present
if ! command -v uv &> /dev/null; then
    echo ""
    echo "📦 Installing UV package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo "✅ UV installed successfully"
else
    echo "✅ UV already installed: $(uv --version)"
fi

echo ""
echo "🔧 Creating virtual environment with UV..."
uv venv .venv --python 3.11

echo ""
echo "📥 Installing all dependencies (this is fast with UV)..."
uv pip install --system -r requirements.txt

echo ""
echo "============================================"
echo "✅ Setup complete!"
echo ""
echo "To run AEGIS SME, open 2 terminals:"
echo ""
echo "  Terminal 1 (API Backend):"
echo "    export PYTHONPATH=."
echo "    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "  Terminal 2 (Dashboard):"
echo "    export PYTHONPATH=."
echo "    python -m streamlit run dashboard/app.py --server.port 8501"
echo ""
echo "  Open browser: http://localhost:8501"
echo "============================================"
