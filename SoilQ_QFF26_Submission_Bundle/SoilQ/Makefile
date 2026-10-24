.PHONY: install audit test lint api dashboard
install:
	pip install -e ".[dev]"
audit:
	python scripts/audit_dataset.py --input data/raw --output data/reports/audit.json
test:
	pytest
lint:
	ruff check src app scripts tests
api:
	uvicorn soilq.api:app --reload
dashboard:
	streamlit run app/dashboard.py
