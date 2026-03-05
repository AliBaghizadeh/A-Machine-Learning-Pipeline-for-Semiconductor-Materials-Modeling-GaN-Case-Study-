.PHONY: setup demo-artifacts demo rag-ingest rag-report test

# Local-only demo: no network calls, no external APIs.

PY ?= python
RAG_SOURCES ?= literature

setup:
	@echo "Offline demo setup notes"
	@echo "1) Create conda env: conda env create -f environment.yml && conda activate mlip_env"
	@echo "2) Install GPAW PAW datasets if you plan to run DFT (not required for demo UI): gpaw install-data \$$HOME/gpaw-data"
	@echo "3) Optional: install Streamlit for the demo UI: pip install streamlit"
	@echo "4) Optional: install Ollama locally (for RAG mode). Demo has a no-Ollama fallback."
	@echo ""
	@echo "Smoke check:"
	@$(PY) -c "import sys; print('python=', sys.version.split()[0])"

demo-artifacts:
	@$(PY) scripts/freeze_artifacts.py --mode demo --out-root analysis/artifacts
	@$(PY) scripts/generate_cards.py

demo:
	@$(PY) -m streamlit run app/main.py --server.headless true

rag-ingest:
	@$(PY) -m rag.ingest.ingest_papers --sources $(RAG_SOURCES) --out rag/index
	@$(PY) -m rag.index.build_index --in rag/index --out rag/index

rag-report:
	@$(PY) -m rag.run_agent --index rag/index --demo-data app/demo_data --out analysis/artifacts

test:
	@$(PY) -m pytest -q
