# ============================================================
# AEGIS SME — Makefile
# Team Finvee | Varsity Hackathon 2026
# ============================================================

.PHONY: install run-api run-dashboard dev clean help

install:
	@echo "Checking UV..."
	@echo "Creating virtual environment..."
	uv venv .venv --python python3.12
	@echo "Installing dependencies..."
	uv pip install --python .venv/bin/python -r requirements.txt
	@echo ""
	@echo "Done! Activate venv with: source .venv/bin/activate"

run-api:
	@echo "Starting FastAPI backend on http://localhost:8000 ..."
	PYTHONPATH=. .venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

run-dashboard:
	@echo "Starting Streamlit dashboard on http://localhost:8501 ..."
	PYTHONPATH=. .venv/bin/python -m streamlit run dashboard/app.py --server.port 8501

dev:
	@echo "Starting AEGIS SME (API + Dashboard)..."
	PYTHONPATH=. .venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
	PYTHONPATH=. .venv/bin/python -m streamlit run dashboard/app.py --server.port 8501

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned!"

help:
	@echo ""
	@echo "AEGIS SME - Available Commands"
	@echo "================================"
	@echo "  make install        Install all dependencies with UV into .venv"
	@echo "  make run-api        Start FastAPI backend (port 8000)"
	@echo "  make run-dashboard  Start Streamlit dashboard (port 8501)"
	@echo "  make dev            Start both services (Mac/Linux)"
	@echo "  make clean          Remove cache files"
	@echo ""
