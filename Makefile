.PHONY: run test lint docker-build docker-up docker-down clean

run:
	python -m uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload

test:
	python -m pytest tests/ -v

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

docker-build:
	docker build -f docker/Dockerfile -t shopwise-ecom:latest .

docker-up:
	docker compose -f docker/docker-compose.yml up -d --build

docker-down:
	docker compose -f docker/docker-compose.yml down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
