#!/usr/bin/env python3
"""Benchmark script for comparing memory usage of different Celery task execution approaches.

This script measures the memory consumption of three different approaches:
1. Prefork with multithreading
2. Gevent with multithreading
3. Gevent with gevent pool

The script will run each approach multiple times with different configurations
and generate a report with peak memory usage for each approach.
"""

import os
import sys
import time
import argparse
import subprocess
import gc
from typing import List, Any

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory_profiler import MemoryProfiler, MemoryProfile, plot_memory_profiles
from tasks import (
    prefork_multithreading_task,
    prefork_sequential_task,
    gevent_multithreading_task,
    gevent_pool_task,
)


def check_redis_available() -> bool:
    """Check if Redis is available for use as broker/backend.

    Returns:
        True if Redis is available, False otherwise.
    """
    try:
        import redis

        client = redis.Redis(host="localhost", port=6379, db=0)
        client.ping()
        return True
    except (redis.ConnectionError, ImportError):
        return False


def start_worker(pool: str, concurrency: int = 1) -> subprocess.Popen:
    """Start a Celery worker with the specified pool.

    Args:
        pool: The worker pool to use ('prefork' or 'gevent').
        concurrency: Number of worker processes/greenlets.

    Returns:
        The subprocess handle for the worker.
    """
    # Ensure we're using the virtual environment's Python
    python_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python"
    )

    cmd = [
        python_path,
        "-m",
        "celery",
        "-A",
        "tasks",
        "worker",
        "--pool",
        pool,
        "--concurrency",
        str(concurrency),
        "--loglevel",
        "INFO",
        "--without-heartbeat",  # Disable heartbeat for benchmark
        "--without-gossip",  # Disable gossip for benchmark
        "--without-mingle",  # Disable mingle for benchmark
    ]

    # Start the worker and return the process handle
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    # Wait for the worker to start up
    print("Waiting for worker to start...")
    time.sleep(10)  # Give more time for worker startup
    return process


def run_benchmark(
    task_func: Any,
    pool_type: str,
    num_subtasks: int,
    subtask_size_mb: float,
    subtask_time: float,
    max_workers: int,
    worker_concurrency: int = 1,
    num_runs: int = 3,
    timeout: int = 120,
) -> MemoryProfile:
    """Run a benchmark for a specific task and configuration.

    Args:
        task_func: The Celery task function to benchmark.
        pool_type: The worker pool type ('prefork' or 'gevent').
        num_subtasks: Number of subtasks to execute.
        subtask_size_mb: Size of data to create in each subtask (MB).
        subtask_time: Processing time for each subtask (seconds).
        max_workers: Maximum number of threads/greenlets to use.
        worker_concurrency: Number of worker processes/greenlets.
        num_runs: Number of times to run the benchmark.

    Returns:
        A MemoryProfile object with the benchmark results.
    """
    print(f"Starting {pool_type} worker with {task_func.__name__}...")
    worker_process = start_worker(pool_type, worker_concurrency)

    try:
        # Force garbage collection before starting
        gc.collect()

        # Initialize memory profiler with shorter interval for more detailed tracking
        profiler = MemoryProfiler(interval=0.05)
        profiler.start()

        # Run the task multiple times and take the average
        for i in range(num_runs):
            print(f"Run {i + 1}/{num_runs}...")

            # Force garbage collection before each run
            gc.collect()

            # Execute the task
            result = task_func.delay(
                num_subtasks=num_subtasks,
                subtask_size_mb=subtask_size_mb,
                subtask_time=subtask_time,
                max_workers=max_workers,
            )

            # Wait for the task to complete
            result.get(timeout=timeout)

            # Wait a moment to ensure all memory is accounted for
            time.sleep(0.5)

        # Stop profiling
        profiler.stop()

        # Final garbage collection
        gc.collect()

        # Get the profile with a descriptive name
        profile_name = (
            f"{pool_type}_{task_func.__name__}_{num_subtasks}x{subtask_size_mb}MB"
        )
        profile = profiler.get_profile(name=profile_name)

        return profile

    finally:
        # Terminate the worker process
        worker_process.terminate()
        worker_process.wait()
        time.sleep(2)  # Give some time for cleanup


def run_all_benchmarks(
    num_subtasks: int = 10,
    subtask_size_mb: float = 20.0,
    subtask_time: float = 0.5,
    max_workers: int = 4,
    worker_concurrency: int = 1,
    num_runs: int = 3,
    timeout: int = 120,
) -> List[MemoryProfile]:
    """Run benchmarks for all three approaches.

    Args:
        num_subtasks: Number of subtasks to execute.
        subtask_size_mb: Size of data to create in each subtask (MB).
        subtask_time: Processing time for each subtask (seconds).
        max_workers: Maximum number of threads/greenlets to use.
        worker_concurrency: Number of worker processes/greenlets.
        num_runs: Number of times to run each benchmark.

    Returns:
        A list of MemoryProfile objects with the benchmark results.
    """
    profiles = []

    # 1. Prefork with multithreading
    profile1 = run_benchmark(
        task_func=prefork_multithreading_task,
        pool_type="prefork",
        num_subtasks=num_subtasks,
        subtask_size_mb=subtask_size_mb,
        subtask_time=subtask_time,
        max_workers=max_workers,
        worker_concurrency=worker_concurrency,
        num_runs=num_runs,
        timeout=timeout,
    )
    profiles.append(profile1)

    # 2. Gevent with multithreading
    profile2 = run_benchmark(
        task_func=gevent_multithreading_task,
        pool_type="gevent",
        num_subtasks=num_subtasks,
        subtask_size_mb=subtask_size_mb,
        subtask_time=subtask_time,
        max_workers=max_workers,
        worker_concurrency=worker_concurrency,
        num_runs=num_runs,
        timeout=timeout,
    )
    profiles.append(profile2)

    # 3. Gevent with gevent pool
    profile3 = run_benchmark(
        task_func=gevent_pool_task,
        pool_type="gevent",
        num_subtasks=num_subtasks,
        subtask_size_mb=subtask_size_mb,
        subtask_time=subtask_time,
        max_workers=max_workers,
        worker_concurrency=worker_concurrency,
        num_runs=num_runs,
        timeout=timeout,
    )
    profiles.append(profile3)

    # 4. Prefork with sequential processing
    profile4 = run_benchmark(
        task_func=prefork_sequential_task,
        pool_type="prefork",
        num_subtasks=num_subtasks,
        subtask_size_mb=subtask_size_mb,
        subtask_time=subtask_time,
        max_workers=max_workers,
        worker_concurrency=worker_concurrency,
        num_runs=num_runs,
        timeout=timeout,
    )
    profiles.append(profile4)

    return profiles


def print_results(profiles: List[MemoryProfile]) -> None:
    """Print the benchmark results in a formatted table.

    Args:
        profiles: List of MemoryProfile objects with benchmark results.
    """

    # Extract the task name from the full name for cleaner display
    def get_display_name(full_name: str) -> str:
        parts = full_name.split("_")
        # Get the task name without the pool type prefix
        if len(parts) >= 2:
            task_name = parts[1]
            # Add the data size info
            size_info = parts[-1] if len(parts) > 2 else ""
            return f"{task_name} ({parts[0]}) {size_info}"
        return full_name

    # Find the longest name for proper padding
    max_name_length = max(len(get_display_name(p.name)) for p in profiles) + 2

    # Table headers and formatting
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    # Header row with dynamic width based on content
    header_format = f"{{:<{max_name_length}}}  {{:>15}}  {{:>15}}"
    print(header_format.format("Approach", "Peak Memory (MB)", "Execution Time (s)"))
    print("-" * 80)

    # Sort profiles by peak memory usage for easier comparison
    sorted_profiles = sorted(profiles, key=lambda p: p.peak_memory)

    # Data rows
    row_format = f"{{:<{max_name_length}}}  {{:>15.2f}}  {{:>15.2f}}"
    for profile in sorted_profiles:
        display_name = get_display_name(profile.name)
        print(
            row_format.format(display_name, profile.peak_memory, profile.execution_time)
        )

    print("=" * 80)


def main() -> None:
    """Main function to run the benchmarks."""
    parser = argparse.ArgumentParser(
        description="Benchmark memory usage of different Celery task execution approaches"
    )
    parser.add_argument(
        "--subtasks", type=int, default=10, help="Number of subtasks to execute"
    )
    parser.add_argument(
        "--size",
        type=float,
        default=50.0,
        help="Size of data to create in each subtask (MB)",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=0.5,
        help="Processing time for each subtask (seconds)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Maximum number of threads/greenlets to use",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Number of worker processes/greenlets",
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="Number of times to run each benchmark"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Timeout for task completion in seconds",
    )

    args = parser.parse_args()

    # Check if Redis is available
    if not check_redis_available():
        print("\nERROR: Redis is not available. Please install and start Redis server:")
        print("  sudo apt-get install redis-server")
        print("  sudo systemctl start redis-server")
        print(
            "\nAlternatively, you can modify celery_app.py to use a different broker/backend."
        )
        return

    print("Starting memory benchmarks with the following configuration:")
    print(f"  Number of subtasks: {args.subtasks}")
    print(f"  Subtask data size: {args.size} MB")
    print(f"  Subtask processing time: {args.time} seconds")
    print(f"  Max workers (threads/greenlets): {args.workers}")
    print(f"  Worker concurrency: {args.concurrency}")
    print(f"  Number of runs per benchmark: {args.runs}")
    print(f"  Task timeout: {args.timeout} seconds")
    print("\nThis will benchmark four approaches:")
    print("  1. Prefork with multithreading")
    print("  2. Prefork with sequential processing")
    print("  3. Gevent with multithreading")
    print("  4. Gevent with gevent pool")
    print("\nRunning benchmarks...\n")

    try:
        # Run all benchmarks
        profiles = run_all_benchmarks(
            num_subtasks=args.subtasks,
            subtask_size_mb=args.size,
            subtask_time=args.time,
            max_workers=args.workers,
            worker_concurrency=args.concurrency,
            num_runs=args.runs,
            timeout=args.timeout,
        )

        # Print results
        print_results(profiles)

        # Plot results
        plot_memory_profiles(
            profiles, title="Memory Usage Comparison of Celery Task Approaches"
        )
        print("\nPlot saved as 'memory_comparison.png'")
    except Exception as e:
        print(f"\nERROR: Benchmark failed: {str(e)}")
        print("\nTroubleshooting tips:")
        print("  1. Make sure Redis server is running")
        print("  2. Try reducing the number of subtasks or their size")
        print("  3. Increase the task timeout with --timeout")
        print("  4. Check that all dependencies are installed correctly")
        import traceback

        print(f"\nError details:\n{traceback.format_exc()}")
        return


if __name__ == "__main__":
    main()
