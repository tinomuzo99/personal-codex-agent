.PHONY: setup run reindex fmt

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

run:
	streamlit run app/main.py

reindex:
	python -m app.rag --reindex

fmt:
	python -m pip install ruff black && ruff check --fix . || true && black . || true