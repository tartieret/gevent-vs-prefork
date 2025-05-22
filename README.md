# Celery Memory Benchmark: Gevent vs Threads

This project benchmarks memory consumption of different Celery task execution approaches, specifically comparing:

1. **Prefork with multithreading**: Standard Celery prefork pool with tasks using Python's threading
2. **Prefork with sequential processing**: Standard Celery prefork pool with tasks processed sequentially (no threading)
3. **Gevent with multithreading**: Celery gevent pool with tasks using Python's threading
4. **Gevent with gevent pool**: Celery gevent pool with tasks using gevent pools for concurrency

## Project Structure

- `run.py`: Main benchmark script
- `tasks.py`: Celery task implementations for each approach
- `celery_app.py`: Celery application configuration
- `memory_profiler.py`: Utilities for measuring memory usage
- `requirements.txt`: Project dependencies

## Installation

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Benchmark

The benchmark script accepts several parameters to customize the test:

```bash
python run.py [options]
```

### Options

- `--subtasks`: Number of subtasks to execute (default: 10)
- `--size`: Size of data to create in each subtask in MB (default: 20.0)
- `--time`: Processing time for each subtask in seconds (default: 0.5)
- `--workers`: Maximum number of threads/greenlets to use (default: 4)
- `--concurrency`: Number of worker processes/greenlets (default: 1)
- `--runs`: Number of times to run each benchmark (default: 3)

### Example

```bash
python run.py --subtasks 15 --size 30 --workers 8
```

## Results

The benchmark will:

1. Print a table with peak memory usage and execution time for each approach
2. Generate a plot (`memory_comparison.png`) showing memory usage over time

## How It Works

Each benchmark:

1. Starts a Celery worker with the specified pool type
2. Runs a task that creates and processes large data objects using subtasks
3. Measures memory consumption throughout execution
4. Records peak memory usage and execution time

The benchmark simulates real-world scenarios where Celery tasks generate large data objects in subtasks, allowing you to compare the memory efficiency of different concurrency approaches.
