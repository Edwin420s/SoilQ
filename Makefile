.PHONY: install prepare audit train experiment test lint api dashboard notebooks challenges clean

install:
	pip install -e ".[dev]"

prepare:
	python scripts/prepare_data.py

audit:
	python scripts/audit_dataset.py --input data/raw --output data/reports/audit.json

train:
	python scripts/train.py \
	  --input data/processed/kenyan_maize_merged.csv \
	  --target high_yield \
	  --features Height_6th Height_10th SPAD_6th SPAD_10th "Stover_Mg/ha" \
	  --positive-label 1 \
	  --output-dir models/experiment_001 \
	  --report data/reports/experiment_001.json \
	  --quantum-maxiter 60

experiment:
	python scripts/run_experiment.py --quantum-maxiter 40

notebooks:
	python scripts/generate_notebooks.py

challenges:
	python ../Qiskit-Fall-Fest-Nairobi-2026-Hackathon-Prompts/solutions/run_all_challenges.py

test:
	pytest -v

lint:
	ruff check src app scripts tests

api:
	uvicorn soilq.api:app --reload

dashboard:
	streamlit run app/dashboard.py

clean:
	rm -rf models/experiment_001 models/dashboard_run
	rm -f data/reports/experiment_001.json data/reports/dashboard_experiment.json
