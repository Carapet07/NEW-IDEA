"""
📊 Logging Setup Module
Centralized logging configuration for the ML training pipeline.

This module provides:
- Structured logging for training events
- Performance monitoring
- Error tracking and debugging
- Training metrics collection
- File and console output management
"""

import logging
import logging.handlers
import datetime
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json


class TrainingLogger:
    """
    Centralized logging system for ML training pipeline.
    
    Features:
    - Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - File rotation to prevent large log files
    - Structured logging with JSON format for metrics
    - Console and file output
    - Training session tracking
    """
    
    def __init__(
        self, 
        log_dir: str = "logs", 
        session_name: Optional[str] = None,
        log_level: str = "INFO",
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Generate session name if not provided
        if session_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            session_name = f"training_session_{timestamp}"
        
        self.session_name = session_name
        self.session_start_time = datetime.datetime.now()
        
        # Setup loggers
        self.setup_main_logger(log_level, max_file_size, backup_count)
        self.setup_metrics_logger()
        self.setup_performance_logger()
        
        # Session tracking
        self.training_metrics = {
            'session_name': session_name,
            'start_time': self.session_start_time.isoformat(),
            'episodes': [],
            'model_saves': [],
            'errors': []
        }
        
        self.log_session_start()
    
    def setup_main_logger(self, log_level: str, max_file_size: int, backup_count: int) -> None:
        """Setup the main application logger."""
        # Create main logger
        self.logger = logging.getLogger('escape_cage_training')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # File handler with rotation
        log_file = self.log_dir / f"{self.session_name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Formatters
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler.setFormatter(console_formatter)
        file_handler.setFormatter(file_formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def setup_metrics_logger(self) -> None:
        """Setup logger for training metrics in JSON format."""
        self.metrics_logger = logging.getLogger('training_metrics')
        self.metrics_logger.setLevel(logging.INFO)
        self.metrics_logger.handlers.clear()
        
        # Metrics file handler
        metrics_file = self.log_dir / f"{self.session_name}_metrics.jsonl"
        metrics_handler = logging.FileHandler(metrics_file, encoding='utf-8')
        
        # Simple formatter for JSON
        metrics_formatter = logging.Formatter('%(message)s')
        metrics_handler.setFormatter(metrics_formatter)
        
        self.metrics_logger.addHandler(metrics_handler)
    
    def setup_performance_logger(self) -> None:
        """Setup logger for performance monitoring."""
        self.perf_logger = logging.getLogger('performance')
        self.perf_logger.setLevel(logging.INFO)
        self.perf_logger.handlers.clear()
        
        # Performance file handler
        perf_file = self.log_dir / f"{self.session_name}_performance.log"
        perf_handler = logging.FileHandler(perf_file, encoding='utf-8')
        
        perf_formatter = logging.Formatter(
            '%(asctime)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        perf_handler.setFormatter(perf_formatter)
        
        self.perf_logger.addHandler(perf_handler)
    
    def log_session_start(self) -> None:
        """Log the start of a training session."""
        self.logger.info("=" * 60)
        self.logger.info(f"🚀 Starting training session: {self.session_name}")
        self.logger.info(f"⏰ Session start time: {self.session_start_time}")
        self.logger.info(f"📁 Log directory: {self.log_dir}")
        self.logger.info("=" * 60)
    
    def log_episode(
        self, 
        episode_num: int, 
        steps: int, 
        reward: float, 
        success: bool,
        additional_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log episode completion with metrics.
        
        Args:
            episode_num: Episode number
            steps: Number of steps in episode
            reward: Total episode reward
            success: Whether episode was successful (escaped)
            additional_metrics: Additional metrics to log
        """
        # Log to main logger
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(
            f"Episode {episode_num:4d} | {status:7s} | "
            f"Steps: {steps:3d} | Reward: {reward:7.2f}"
        )
        
        # Prepare metrics
        episode_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'episode': episode_num,
            'steps': steps,
            'reward': reward,
            'success': success,
            'duration_seconds': None  # Can be added if timing is tracked
        }
        
        if additional_metrics:
            episode_data.update(additional_metrics)
        
        # Log to metrics file
        self.metrics_logger.info(json.dumps(episode_data))
        
        # Add to session tracking
        self.training_metrics['episodes'].append(episode_data)
    
    def log_model_save(self, model_name: str, episode: int, performance_metrics: Optional[Dict] = None) -> None:
        """Log model saving event."""
        self.logger.info(f"💾 Model saved: {model_name} (Episode {episode})")
        
        save_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'model_name': model_name,
            'episode': episode,
            'performance_metrics': performance_metrics or {}
        }
        
        self.training_metrics['model_saves'].append(save_data)
    
    def log_error(self, error_msg: str, error_type: str = "GENERAL", exception: Optional[Exception] = None) -> None:
        """Log error with context."""
        error_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'error_type': error_type,
            'message': error_msg,
            'exception': str(exception) if exception else None
        }
        
        self.logger.error(f"❌ {error_type}: {error_msg}")
        if exception:
            self.logger.error(f"Exception details: {exception}")
        
        self.training_metrics['errors'].append(error_data)
    
    def log_performance(self, operation: str, duration: float, additional_data: Optional[Dict] = None) -> None:
        """Log performance metrics."""
        perf_msg = f"{operation}: {duration:.4f}s"
        if additional_data:
            extras = ", ".join([f"{k}={v}" for k, v in additional_data.items()])
            perf_msg += f" | {extras}"
        
        self.perf_logger.info(perf_msg)
    
    def log_training_progress(self, current_step: int, total_steps: int, fps: Optional[float] = None) -> None:
        """Log training progress."""
        progress = (current_step / total_steps) * 100
        elapsed = datetime.datetime.now() - self.session_start_time
        
        msg = f"Training Progress: {progress:5.1f}% ({current_step}/{total_steps}) | Elapsed: {elapsed}"
        if fps:
            msg += f" | FPS: {fps:.1f}"
        
        self.logger.info(msg)
    
    def save_session_summary(self) -> None:
        """Save complete session summary to file."""
        self.training_metrics['end_time'] = datetime.datetime.now().isoformat()
        self.training_metrics['total_duration'] = str(datetime.datetime.now() - self.session_start_time)
        
        # Calculate summary statistics
        episodes = self.training_metrics['episodes']
        if episodes:
            successful_episodes = [ep for ep in episodes if ep['success']]
            self.training_metrics['summary'] = {
                'total_episodes': len(episodes),
                'successful_episodes': len(successful_episodes),
                'success_rate': len(successful_episodes) / len(episodes) if episodes else 0,
                'average_reward': sum(ep['reward'] for ep in episodes) / len(episodes),
                'average_steps': sum(ep['steps'] for ep in episodes) / len(episodes),
                'total_errors': len(self.training_metrics['errors']),
                'total_model_saves': len(self.training_metrics['model_saves'])
            }
        
        # Save to file
        summary_file = self.log_dir / f"{self.session_name}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.training_metrics, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📊 Session summary saved: {summary_file}")
    
    def close(self) -> None:
        """Close logger and save final summary."""
        self.logger.info("=" * 60)
        self.logger.info(f"🏁 Training session ended: {self.session_name}")
        self.logger.info(f"⏱️ Total duration: {datetime.datetime.now() - self.session_start_time}")
        self.logger.info("=" * 60)
        
        self.save_session_summary()
        
        # Close handlers
        for handler in self.logger.handlers:
            handler.close()


def setup_logging(
    log_dir: str = "logs",
    session_name: Optional[str] = None,
    log_level: str = "INFO"
) -> TrainingLogger:
    """
    Quick setup function for training logging.
    
    Args:
        log_dir: Directory for log files
        session_name: Name for this training session
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured TrainingLogger instance
    """
    return TrainingLogger(log_dir, session_name, log_level)


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module."""
    return logging.getLogger(name)


# Performance monitoring decorators
def log_performance(operation_name: str):
    """Decorator to log function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.datetime.now()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.datetime.now() - start_time).total_seconds()
                
                perf_logger = logging.getLogger('performance')
                perf_logger.info(f"{operation_name}: {duration:.4f}s")
                
                return result
            except Exception as e:
                duration = (datetime.datetime.now() - start_time).total_seconds()
                error_logger = logging.getLogger('escape_cage_training')
                error_logger.error(f"{operation_name} failed after {duration:.4f}s: {e}")
                raise
        return wrapper
    return decorator 