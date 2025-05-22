.PHONY: start-redis stop-redis install run clean help

# Redis container settings
REDIS_CONTAINER_NAME = celery-redis
REDIS_PORT = 6379

help:
	@echo "Available commands:"
	@echo "  make install         - Install dependencies in virtual environment"
	@echo "  make start-redis     - Start Redis container for benchmarking"
	@echo "  make stop-redis      - Stop and remove Redis container"
	@echo "  make run             - Run benchmark with default settings"
	@echo "  make run-small       - Run benchmark with small data size"
	@echo "  make run-large       - Run benchmark with large data size"
	@echo "  make run-reallife    - Run benchmark matching the use-case for ioTORQ LEAN"
	@echo "  make clean           - Clean up generated files"

install:
	@echo "Creating virtual environment and installing dependencies..."
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

start-redis:
	@echo "Starting Redis container..."
	docker run --name $(REDIS_CONTAINER_NAME) -d -p $(REDIS_PORT):6379 redis:alpine
	@echo "Redis is running on localhost:$(REDIS_PORT)"

stop-redis:
	@echo "Stopping Redis container..."
	docker stop $(REDIS_CONTAINER_NAME) || true
	docker rm $(REDIS_CONTAINER_NAME) || true

run: 
	@echo "Running benchmark with default settings..."
	. .venv/bin/activate && python run.py

run-small:
	@echo "Running benchmark with small data size..."
	. .venv/bin/activate && python run.py --subtasks 5 --size 10 --workers 4 --runs 3

run-large:
	@echo "Running benchmark with large data size..."
	. .venv/bin/activate && python run.py --subtasks 15 --size 50 --workers 8 --runs 5

run-reallife:
	@echo "Running benchmark matching the use-case for ioTORQ LEAN..."
	. .venv/bin/activate && python run.py --subtasks 200 --size 3 --workers 40 --runs 3

clean:
	@echo "Cleaning up generated files..."
	rm -f memory_comparison.png
	rm -rf __pycache__
	rm -rf .pytest_cache
