# ---------------------------------------------------------------------------
# Convenience wrappers around the Docker-Compose commands you'd otherwise type.
# Use the underlying `docker compose ...` commands directly if `make` isn't
# installed on your system (common on Windows).
#
# Examples:
#   make build
#   make ingest PDF=data/sample.pdf
#   make ask Q="What is the refund policy?"
#   make ui
#   make clear
#   make clean
#   make test
# ---------------------------------------------------------------------------

PDF ?= data/sample.pdf
Q   ?= What is this document about?

.PHONY: help build ingest ask ui clear clean test

help:
	@echo "Targets:"
	@echo "  build     Build the Docker image"
	@echo "  ingest    Ingest a PDF (or folder)  e.g.  make ingest PDF=data/my.pdf"
	@echo "  ask       Ask a question            e.g.  make ask Q=\"...\""
	@echo "  ui        Launch the Streamlit chat UI on http://localhost:8501"
	@echo "  clear     Wipe the vector store collection (keeps the folder)"
	@echo "  clean     Remove the on-disk chroma_db/ folder entirely"
	@echo "  test      Run the unit tests with pytest"

build:
	docker compose build

ingest:
	docker compose run --rm rag python -m src.cli ingest $(PDF)

ask:
	docker compose run --rm rag python -m src.cli ask "$(Q)"

ui:
	docker compose up ui

clear:
	docker compose run --rm rag python -m src.cli clear --yes

clean:
	rm -rf chroma_db

test:
	docker compose run --rm rag python -m pytest -q
