.PHONY: install data train test lint api docker
install:
	python -m pip install -r requirements-dev.txt
data:
	python -m src.data --output data/heart_disease.csv
train: data
	python -m src.modeling --data data/heart_disease.csv --model-dir artifacts --tracking-uri ./mlruns
test:
	python -m pytest -q
lint:
	ruff check .
api:
	uvicorn src.api:app --host 127.0.0.1 --port 8000
docker:
	docker build -t heart-disease-api:local .
