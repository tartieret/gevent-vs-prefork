"""
Celery application configuration.
"""
from typing import Dict, Any, Optional
from celery import Celery


def create_app(
    broker: str = "redis://localhost:6379/0",
    backend: str = "redis://localhost:6379/0",
    worker_pool: Optional[str] = None,
    **kwargs: Any
) -> Celery:
    """
    Create a Celery application with the specified configuration.
    
    Args:
        broker: The broker URL to use.
        backend: The result backend URL to use.
        worker_pool: The worker pool implementation to use (e.g., 'prefork', 'gevent').
        **kwargs: Additional configuration options for Celery.
        
    Returns:
        A configured Celery application.
    """
    app = Celery(
        "memory_benchmark",
        broker=broker,
        backend=backend,
    )
    
    # Default configuration
    config: Dict[str, Any] = {
        "task_serializer": "pickle",
        "accept_content": ["pickle", "json"],
        "result_serializer": "pickle",
        "task_acks_late": True,
        "worker_prefetch_multiplier": 1,
        "worker_concurrency": 4,
    }
    
    # Apply worker pool if specified
    if worker_pool:
        config["worker_pool"] = worker_pool
    
    # Apply any additional configuration
    config.update(kwargs)
    
    app.conf.update(config)
    return app
