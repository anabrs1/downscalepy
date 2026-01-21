"""
Performance monitoring utilities for bias correction solver.
"""

import time
import psutil
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Monitor performance metrics during solver execution.
    
    Tracks:
    - Execution time
    - Memory usage
    - Function call counts
    - Convergence metrics
    """
    
    def __init__(self):
        self.metrics = {
            'start_time': None,
            'end_time': None,
            'elapsed_time': None,
            'peak_memory_mb': 0,
            'function_calls': {},
            'convergence': []
        }
        self.process = psutil.Process()
        self._timers = {}
    
    def start(self):
        """Start monitoring."""
        self.metrics['start_time'] = datetime.now()
        self._update_memory()
        logger.info("Performance monitoring started")
    
    def stop(self):
        """Stop monitoring and calculate final metrics."""
        self.metrics['end_time'] = datetime.now()
        if self.metrics['start_time']:
            elapsed = self.metrics['end_time'] - self.metrics['start_time']
            self.metrics['elapsed_time'] = elapsed.total_seconds()
        self._update_memory()
        logger.info(f"Performance monitoring stopped. Elapsed: {self.metrics['elapsed_time']:.2f}s")
    
    def _update_memory(self):
        """Update peak memory usage."""
        mem_info = self.process.memory_info()
        current_mb = mem_info.rss / 1024 / 1024
        self.metrics['peak_memory_mb'] = max(
            self.metrics['peak_memory_mb'],
            current_mb
        )
    
    @contextmanager
    def timer(self, name: str):
        """
        Context manager for timing code blocks.
        
        Usage:
            with monitor.timer('optimization'):
                # code to time
        """
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            if name not in self._timers:
                self._timers[name] = []
            self._timers[name].append(elapsed)
            self._update_memory()
    
    def record_call(self, function_name: str):
        """Record a function call."""
        if function_name not in self.metrics['function_calls']:
            self.metrics['function_calls'][function_name] = 0
        self.metrics['function_calls'][function_name] += 1
    
    def record_convergence(self, lu_from: int, convergence_info: Dict[str, Any]):
        """Record convergence information."""
        self.metrics['convergence'].append({
            'lu_from': lu_from,
            **convergence_info
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {
            'elapsed_time_seconds': self.metrics['elapsed_time'],
            'peak_memory_mb': self.metrics['peak_memory_mb'],
            'total_function_calls': sum(self.metrics['function_calls'].values()),
            'convergence_success_rate': self._calculate_success_rate()
        }
        
        # Add timing summaries
        if self._timers:
            summary['timings'] = {
                name: {
                    'total': sum(times),
                    'mean': sum(times) / len(times),
                    'count': len(times)
                }
                for name, times in self._timers.items()
            }
        
        return summary
    
    def _calculate_success_rate(self) -> float:
        """Calculate convergence success rate."""
        if not self.metrics['convergence']:
            return 0.0
        successful = sum(
            1 for c in self.metrics['convergence']
            if c.get('success', False)
        )
        return successful / len(self.metrics['convergence'])
    
    def print_summary(self):
        """Print formatted performance summary."""
        summary = self.get_summary()
        
        print("\n" + "="*70)
        print("PERFORMANCE SUMMARY")
        print("="*70)
        print(f"Elapsed Time: {summary['elapsed_time_seconds']:.2f} seconds")
        print(f"Peak Memory: {summary['peak_memory_mb']:.2f} MB")
        print(f"Convergence Success Rate: {summary['convergence_success_rate']:.1%}")
        
        if 'timings' in summary:
            print("\nTiming Breakdown:")
            for name, timing in summary['timings'].items():
                print(f"  {name}: {timing['total']:.2f}s (mean: {timing['mean']:.3f}s, n={timing['count']})")
        
        print("="*70 + "\n")
