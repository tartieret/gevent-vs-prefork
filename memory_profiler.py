"""
Utility for measuring memory usage during execution.
"""
from typing import Callable, Any, List, Optional
import time
import os
import psutil
import threading
from functools import wraps
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class MemoryProfile:
    """Data class to store memory profiling results."""
    timestamps: List[float]
    memory_usage: List[float]
    peak_memory: float
    execution_time: float
    name: str


class MemoryProfiler:
    """
    A class to profile memory usage of a function or process.
    
    This profiler tracks memory usage over time and records peak memory consumption.
    """
    
    def __init__(self, interval: float = 0.1):
        """
        Initialize the memory profiler.
        
        Args:
            interval: Time interval in seconds between memory measurements.
        """
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.stop_flag = False
        self.memory_usage: List[float] = []
        self.timestamps: List[float] = []
        self.start_time: float = 0
        self.end_time: float = 0
        self.profiling_thread: Optional[threading.Thread] = None
    
    def _profile_memory(self) -> None:
        """Background thread function to measure memory usage at regular intervals."""
        while not self.stop_flag:
            # Get memory for the current process
            mem = self.process.memory_info().rss / (1024 * 1024)  # Convert to MB
            
            # Get memory for all child processes
            try:
                children = self.process.children(recursive=True)
                for child in children:
                    try:
                        child_mem = child.memory_info().rss / (1024 * 1024)
                        mem += child_mem  # Add child memory to total
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass  # Process might have terminated
            except (AttributeError, psutil.NoSuchProcess):
                pass  # Process might not have children method or might have terminated
            
            self.memory_usage.append(mem)
            self.timestamps.append(time.time() - self.start_time)
            time.sleep(self.interval)
    
    def start(self) -> None:
        """Start memory profiling in a background thread."""
        self.stop_flag = False
        self.memory_usage = []
        self.timestamps = []
        self.start_time = time.time()
        
        self.profiling_thread = threading.Thread(target=self._profile_memory)
        self.profiling_thread.daemon = True
        self.profiling_thread.start()
    
    def stop(self) -> None:
        """Stop memory profiling."""
        self.stop_flag = True
        self.end_time = time.time()
        if self.profiling_thread:
            self.profiling_thread.join(timeout=2.0)
    
    def get_profile(self, name: str = "unnamed") -> MemoryProfile:
        """
        Get the memory profile results.
        
        Args:
            name: A name to identify this profile.
            
        Returns:
            A MemoryProfile object containing the profiling data.
        """
        peak_memory = max(self.memory_usage) if self.memory_usage else 0
        execution_time = self.end_time - self.start_time
        
        return MemoryProfile(
            timestamps=self.timestamps,
            memory_usage=self.memory_usage,
            peak_memory=peak_memory,
            execution_time=execution_time,
            name=name
        )


def profile_memory(func: Callable) -> Callable:
    """
    Decorator to profile memory usage of a function.
    
    Args:
        func: The function to profile.
        
    Returns:
        A wrapped function that profiles memory usage.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> tuple[Any, MemoryProfile]:
        profiler = MemoryProfiler()
        profiler.start()
        
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.stop()
        
        profile = profiler.get_profile(name=func.__name__)
        return result, profile
    
    return wrapper


def plot_memory_profiles(profiles: List[MemoryProfile], title: str = "Memory Usage Comparison") -> None:
    """
    Plot memory usage profiles for comparison.
    
    Args:
        profiles: List of MemoryProfile objects to compare.
        title: Title for the plot.
    """
    plt.figure(figsize=(12, 6))
    
    for profile in profiles:
        plt.plot(profile.timestamps, profile.memory_usage, label=f"{profile.name} (Peak: {profile.peak_memory:.2f} MB)")
    
    plt.xlabel("Time (seconds)")
    plt.ylabel("Memory Usage (MB)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # Add peak memory comparison as text
    peak_text = "\n".join([f"{p.name}: {p.peak_memory:.2f} MB (Time: {p.execution_time:.2f}s)" for p in profiles])
    plt.figtext(0.02, 0.02, f"Peak Memory Usage:\n{peak_text}", fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("memory_comparison.png")
    plt.close()
