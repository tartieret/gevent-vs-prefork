"""
Task implementations for memory benchmarking.

This module contains three different implementations of a task that processes
subtasks, each using a different concurrency approach:
1. Prefork with multithreading
2. Gevent with multithreading
3. Gevent with gevent pool
"""

from typing import List, Dict, Any
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import gevent
from gevent.pool import Pool as GeventPool

from celery_app import create_app


# Create the Celery app instances for each approach
prefork_app = create_app(worker_pool="prefork")
prefork_sequential_app = create_app(worker_pool="prefork")
gevent_app = create_app(worker_pool="gevent")
gevent_pool_app = create_app(worker_pool="gevent")


def create_large_data(size_mb: float = 10.0) -> np.ndarray:
    """
    Create a large numpy array to simulate memory-intensive operations.

    Args:
        size_mb: Size of the array in megabytes.

    Returns:
        A numpy array of the specified size.
    """
    # Calculate number of elements needed for the given size in MB
    # Each float64 is 8 bytes
    num_elements = int((size_mb * 1024 * 1024) / 8)
    return np.random.random(num_elements)


def process_subtask(
    task_id: int, size_mb: float = 10.0, processing_time: float = 1.0
) -> Dict[str, Any]:
    """
    Process a single subtask that creates and manipulates large data.

    Args:
        task_id: Identifier for this subtask.
        size_mb: Size of the data to create in megabytes.
        processing_time: Simulated processing time in seconds.

    Returns:
        A dictionary with task results.
    """
    # Create large data
    data = create_large_data(size_mb)

    # Simulate processing
    time.sleep(processing_time)

    # Do some operations on the data to ensure it's not optimized away
    result = np.mean(data)
    std_dev = np.std(data)

    return {
        "task_id": task_id,
        "data_size_mb": size_mb,
        "mean": result,
        "std_dev": std_dev,
    }


# ===== Approach 1: Prefork with Multithreading =====
@prefork_app.task(name="prefork_multithreading_task")
def prefork_multithreading_task(
    num_subtasks: int = 5,
    subtask_size_mb: float = 10.0,
    subtask_time: float = 1.0,
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    """
    Execute subtasks using multithreading in a prefork worker.

    Args:
        num_subtasks: Number of subtasks to execute.
        subtask_size_mb: Size of data to create in each subtask (MB).
        subtask_time: Processing time for each subtask (seconds).
        max_workers: Maximum number of threads to use.

    Returns:
        List of results from all subtasks.
    """
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_subtask,
                task_id=i,
                size_mb=subtask_size_mb,
                processing_time=subtask_time,
            )
            for i in range(num_subtasks)
        ]

        for future in futures:
            results.append(future.result())

    return results


# ===== Approach 2: Gevent with Multithreading =====
@gevent_app.task(name="gevent_multithreading_task")
def gevent_multithreading_task(
    num_subtasks: int = 5,
    subtask_size_mb: float = 10.0,
    subtask_time: float = 1.0,
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    """
    Execute subtasks using multithreading in a gevent worker.

    Args:
        num_subtasks: Number of subtasks to execute.
        subtask_size_mb: Size of data to create in each subtask (MB).
        subtask_time: Processing time for each subtask (seconds).
        max_workers: Maximum number of threads to use.

    Returns:
        List of results from all subtasks.
    """
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_subtask,
                task_id=i,
                size_mb=subtask_size_mb,
                processing_time=subtask_time,
            )
            for i in range(num_subtasks)
        ]

        for future in futures:
            results.append(future.result())

    return results


# ===== Approach 3: Gevent with Gevent Pool =====
@gevent_pool_app.task(name="gevent_pool_task")
def gevent_pool_task(
    num_subtasks: int = 5,
    subtask_size_mb: float = 10.0,
    subtask_time: float = 1.0,
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    """
    Execute subtasks using a gevent pool in a gevent worker.

    Args:
        num_subtasks: Number of subtasks to execute.
        subtask_size_mb: Size of data to create in each subtask (MB).
        subtask_time: Processing time for each subtask (seconds).
        max_workers: Maximum number of greenlets to use.

    Returns:
        List of results from all subtasks.
    """
    pool = GeventPool(max_workers)
    jobs = [
        pool.spawn(
            process_subtask,
            task_id=i,
            size_mb=subtask_size_mb,
            processing_time=subtask_time,
        )
        for i in range(num_subtasks)
    ]

    # Wait for all jobs to complete
    gevent.joinall(jobs)

    # Get results
    results = [job.value for job in jobs]
    return results


# ===== Approach 4: Prefork with Sequential Processing =====
@prefork_sequential_app.task(name="prefork_sequential_task")
def prefork_sequential_task(
    num_subtasks: int = 5,
    subtask_size_mb: float = 10.0,
    subtask_time: float = 1.0,
    max_workers: int = 4,  # Not used but kept for API consistency
) -> List[Dict[str, Any]]:
    """
    Execute subtasks sequentially (no threading) in a prefork worker.

    Args:
        num_subtasks: Number of subtasks to execute.
        subtask_size_mb: Size of data to create in each subtask (MB).
        subtask_time: Processing time for each subtask (seconds).
        max_workers: Not used, kept for API consistency.

    Returns:
        List of results from all subtasks.
    """
    results = []

    # Process each subtask sequentially in a for loop
    for i in range(num_subtasks):
        result = process_subtask(
            task_id=i, size_mb=subtask_size_mb, processing_time=subtask_time
        )
        results.append(result)

    return results
