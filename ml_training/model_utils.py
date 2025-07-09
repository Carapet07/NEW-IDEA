"""
🔧 Model Utilities Module
Comprehensive utilities for model management, validation, and operations.

This module provides:
- Model loading and saving utilities
- Model validation and integrity checks
- Model comparison and benchmarking
- Model metadata management
- Model backup and versioning utilities
"""

import os
import shutil
import json
import zipfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import datetime
import hashlib
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
import numpy as np
import logging


class ModelMetadata:
    """
    Class to manage model metadata and information.
    
    Stores comprehensive information about a model including:
    - Training configuration and hyperparameters
    - Performance metrics and validation results
    - Creation and modification timestamps
    - Model versioning and backup information
    """
    
    def __init__(self, model_name: str):
        """
        Initialize model metadata.
        
        Args:
            model_name: Name of the model
        """
        self.model_name = model_name
        self.creation_time = datetime.datetime.now().isoformat()
        self.last_modified = self.creation_time
        self.version = "1.0.0"
        self.algorithm = "PPO"
        self.hyperparameters = {}
        self.training_stats = {}
        self.validation_results = {}
        self.file_info = {}
        self.notes = ""
        self.tags = []
    
    def update_training_stats(self, stats: Dict[str, Any]) -> None:
        """Update training statistics."""
        self.training_stats.update(stats)
        self.last_modified = datetime.datetime.now().isoformat()
    
    def update_validation_results(self, results: Dict[str, Any]) -> None:
        """Update validation results."""
        self.validation_results.update(results)
        self.last_modified = datetime.datetime.now().isoformat()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the model."""
        if tag not in self.tags:
            self.tags.append(tag)
            self.last_modified = datetime.datetime.now().isoformat()
    
    def set_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """Set model hyperparameters."""
        self.hyperparameters = hyperparams.copy()
        self.last_modified = datetime.datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'model_name': self.model_name,
            'creation_time': self.creation_time,
            'last_modified': self.last_modified,
            'version': self.version,
            'algorithm': self.algorithm,
            'hyperparameters': self.hyperparameters,
            'training_stats': self.training_stats,
            'validation_results': self.validation_results,
            'file_info': self.file_info,
            'notes': self.notes,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create ModelMetadata from dictionary."""
        metadata = cls(data['model_name'])
        metadata.creation_time = data.get('creation_time', metadata.creation_time)
        metadata.last_modified = data.get('last_modified', metadata.last_modified)
        metadata.version = data.get('version', metadata.version)
        metadata.algorithm = data.get('algorithm', metadata.algorithm)
        metadata.hyperparameters = data.get('hyperparameters', {})
        metadata.training_stats = data.get('training_stats', {})
        metadata.validation_results = data.get('validation_results', {})
        metadata.file_info = data.get('file_info', {})
        metadata.notes = data.get('notes', "")
        metadata.tags = data.get('tags', [])
        return metadata


class ModelValidator:
    """
    Utility class for validating model files and integrity.
    
    Provides comprehensive validation including:
    - File integrity checks
    - Model loading validation
    - Performance validation
    - Configuration validation
    """
    
    def __init__(self):
        self.logger = logging.getLogger('model_validator')
    
    def validate_model_file(self, model_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Comprehensive validation of a model file.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Dictionary containing validation results
        """
        model_path = Path(model_path)
        validation_result = {
            'valid': False,
            'file_exists': False,
            'file_readable': False,
            'model_loadable': False,
            'file_size': 0,
            'file_hash': None,
            'errors': [],
            'warnings': [],
            'model_info': {}
        }
        
        try:
            # Check if file exists
            if not model_path.exists():
                validation_result['errors'].append(f"Model file does not exist: {model_path}")
                return validation_result
            
            validation_result['file_exists'] = True
            validation_result['file_size'] = model_path.stat().st_size
            
            # Check if file is readable
            try:
                with open(model_path, 'rb') as f:
                    # Calculate file hash for integrity
                    file_content = f.read()
                    validation_result['file_hash'] = hashlib.md5(file_content).hexdigest()
                validation_result['file_readable'] = True
            except Exception as e:
                validation_result['errors'].append(f"Cannot read model file: {e}")
                return validation_result
            
            # Try to load the model
            try:
                model = PPO.load(str(model_path))
                validation_result['model_loadable'] = True
                
                # Extract model information
                validation_result['model_info'] = {
                    'policy_class': str(type(model.policy)),
                    'observation_space': str(model.observation_space),
                    'action_space': str(model.action_space),
                    'learning_rate': getattr(model, 'learning_rate', 'unknown'),
                    'n_steps': getattr(model, 'n_steps', 'unknown'),
                    'batch_size': getattr(model, 'batch_size', 'unknown')
                }
                
                # Check for common issues
                if validation_result['file_size'] < 1000:  # Very small file
                    validation_result['warnings'].append("Model file is unusually small")
                
                validation_result['valid'] = True
                
            except Exception as e:
                validation_result['errors'].append(f"Cannot load model: {e}")
                return validation_result
                
        except Exception as e:
            validation_result['errors'].append(f"Unexpected error during validation: {e}")
        
        return validation_result
    
    def validate_model_performance(self, model_path: Union[str, Path], 
                                 expected_performance: Dict[str, float]) -> Dict[str, Any]:
        """
        Validate model performance against expected metrics.
        
        Args:
            model_path: Path to the model file
            expected_performance: Dictionary with expected performance metrics
            
        Returns:
            Dictionary containing performance validation results
        """
        # This would typically run the model through a test environment
        # For now, return a placeholder structure
        return {
            'performance_valid': False,
            'meets_expectations': False,
            'performance_metrics': {},
            'recommendations': []
        }


class ModelManager:
    """
    Comprehensive model management utility.
    
    Provides functionality for:
    - Model discovery and listing
    - Model backup and versioning
    - Model comparison and selection
    - Model cleanup and organization
    """
    
    def __init__(self, models_dir: str = "models", backup_dir: str = "model_backups"):
        """
        Initialize model manager.
        
        Args:
            models_dir: Directory containing model files
            backup_dir: Directory for model backups
        """
        self.models_dir = Path(models_dir)
        self.backup_dir = Path(backup_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)
        self.validator = ModelValidator()
        self.metadata: Dict[str, Any] = {}  # Initialize metadata attribute
        self.logger = logging.getLogger('model_manager')
    
    def discover_models(self, include_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Discover all models in the models directory.
        
        Args:
            include_metadata: Whether to include metadata information
            
        Returns:
            List of dictionaries containing model information
        """
        models = []
        
        # Look for .zip files (SB3 format)
        for model_file in self.models_dir.glob("*.zip"):
            model_info = {
                'name': model_file.stem,
                'path': str(model_file),
                'file_size': model_file.stat().st_size,
                'modified_time': datetime.datetime.fromtimestamp(
                    model_file.stat().st_mtime
                ).isoformat(),
                'validation': None,
                'metadata': None
            }
            
            # Validate model
            validation_result = self.validator.validate_model_file(model_file)
            model_info['validation'] = validation_result
            
            # Load metadata if available and requested
            if include_metadata:
                metadata_file = self.models_dir / f"{model_file.stem}_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata_dict = json.load(f)
                        model_info['metadata'] = ModelMetadata.from_dict(metadata_dict)
                    except Exception as e:
                        self.logger.warning(f"Failed to load metadata for {model_file.stem}: {e}")
            
            models.append(model_info)
        
        # Sort by modification time (newest first)
        models.sort(key=lambda x: x['modified_time'], reverse=True)
        
        return models
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information or None if not found
        """
        models = self.discover_models(include_metadata=True)
        for model in models:
            if model['name'] == model_name:
                return model
        return None
    
    def backup_model(self, model_name: str, backup_suffix: str = "") -> Optional[str]:
        """
        Create a backup of a model.
        
        Args:
            model_name: Name of the model to backup
            backup_suffix: Optional suffix for backup name
            
        Returns:
            Path to backup file or None if failed
        """
        model_path = self.models_dir / f"{model_name}.zip"
        if not model_path.exists():
            self.logger.error(f"Model {model_name} not found for backup")
            return None
        
        # Create backup filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{model_name}_backup_{timestamp}"
        if backup_suffix:
            backup_name += f"_{backup_suffix}"
        backup_path = self.backup_dir / f"{backup_name}.zip"
        
        try:
            shutil.copy2(model_path, backup_path)
            
            # Also backup metadata if it exists
            metadata_path = self.models_dir / f"{model_name}_metadata.json"
            if metadata_path.exists():
                backup_metadata_path = self.backup_dir / f"{backup_name}_metadata.json"
                shutil.copy2(metadata_path, backup_metadata_path)
            
            self.logger.info(f"Model {model_name} backed up to {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            self.logger.error(f"Failed to backup model {model_name}: {e}")
            return None
    
    def compare_models(self, model_names: List[str]) -> Dict[str, Any]:
        """
        Compare multiple models based on metadata and validation.
        
        Args:
            model_names: List of model names to compare
            
        Returns:
            Dictionary containing comparison results
        """
        comparison = {
            'models': {},
            'summary': {},
            'recommendations': []
        }
        
        valid_models = []
        
        for model_name in model_names:
            model_info = self.get_model_info(model_name)
            if model_info and model_info['validation']['valid']:
                comparison['models'][model_name] = model_info
                valid_models.append(model_name)
            else:
                comparison['models'][model_name] = {
                    'error': 'Model not found or invalid'
                }
        
        if len(valid_models) < 2:
            comparison['summary']['status'] = 'Insufficient valid models for comparison'
            return comparison
        
        # Compare file sizes
        sizes = {name: comparison['models'][name]['file_size'] 
                for name in valid_models}
        comparison['summary']['largest_model'] = max(sizes, key=sizes.get)
        comparison['summary']['smallest_model'] = min(sizes, key=sizes.get)
        
        # Compare modification times
        times = {name: comparison['models'][name]['modified_time'] 
                for name in valid_models}
        comparison['summary']['newest_model'] = max(times, key=times.get)
        comparison['summary']['oldest_model'] = min(times, key=times.get)
        
        # Compare metadata if available
        models_with_metadata = [name for name in valid_models 
                              if comparison['models'][name].get('metadata')]
        
        if models_with_metadata:
            # Compare training stats
            training_performances = {}
            for name in models_with_metadata:
                metadata = comparison['models'][name]['metadata']
                if hasattr(metadata, 'training_stats') and 'success_rate' in metadata.training_stats:
                    training_performances[name] = metadata.training_stats['success_rate']
            
            if training_performances:
                best_performer = max(training_performances, key=training_performances.get)
                comparison['summary']['best_performing_model'] = best_performer
                comparison['recommendations'].append(
                    f"Based on training stats, {best_performer} has the highest success rate"
                )
        
        comparison['summary']['status'] = 'Comparison completed'
        return comparison
    
    def save_model_metadata(self, model_name: str, metadata: ModelMetadata) -> bool:
        """
        Save metadata for a model.
        
        Args:
            model_name: Name of the model
            metadata: ModelMetadata instance
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            metadata_path = self.models_dir / f"{model_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save metadata for {model_name}: {e}")
            return False
    
    def load_model_metadata(self, model_name: str) -> Optional[ModelMetadata]:
        """
        Load metadata for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelMetadata instance or None if not found
        """
        try:
            metadata_path = self.models_dir / f"{model_name}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata_dict = json.load(f)
                return ModelMetadata.from_dict(metadata_dict)
        except Exception as e:
            self.logger.error(f"Failed to load metadata for {model_name}: {e}")
        return None
    
    def cleanup_old_backups(self, keep_days: int = 30) -> int:
        """
        Clean up old backup files.
        
        Args:
            keep_days: Number of days to keep backups
            
        Returns:
            Number of files deleted
        """
        deleted_count = 0
        cutoff_time = datetime.datetime.now() - datetime.timedelta(days=keep_days)
        
        for backup_file in self.backup_dir.glob("*.zip"):
            if datetime.datetime.fromtimestamp(backup_file.stat().st_mtime) < cutoff_time:
                try:
                    backup_file.unlink()
                    # Also remove corresponding metadata file
                    metadata_file = backup_file.with_suffix('.json')
                    if metadata_file.exists():
                        metadata_file.unlink()
                    deleted_count += 1
                    self.logger.info(f"Deleted old backup: {backup_file}")
                except Exception as e:
                    self.logger.error(f"Failed to delete backup {backup_file}: {e}")
        
        return deleted_count
    
    def export_model_info(self, model_name: str, export_path: str) -> bool:
        """
        Export comprehensive model information to JSON file.
        
        Args:
            model_name: Name of the model
            export_path: Path to save the export file
            
        Returns:
            True if exported successfully, False otherwise
        """
        model_info = self.get_model_info(model_name)
        if not model_info:
            return False
        
        try:
            # Convert ModelMetadata to dict if present
            export_data = model_info.copy()
            if export_data.get('metadata') and hasattr(export_data['metadata'], 'to_dict'):
                export_data['metadata'] = export_data['metadata'].to_dict()
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to export model info for {model_name}: {e}")
            return False


# Utility functions for common model operations

def find_best_model(models_dir: str = "models", 
                   criteria: str = "newest") -> Optional[str]:
    """
    Find the best model based on specified criteria.
    
    Args:
        models_dir: Directory containing models
        criteria: Selection criteria ("newest", "largest", "smallest")
        
    Returns:
        Name of the best model or None if no models found
    """
    manager = ModelManager(models_dir)
    models = manager.discover_models(include_metadata=True)
    
    if not models:
        return None
    
    if criteria == "newest":
        return max(models, key=lambda x: x['modified_time'])['name']
    elif criteria == "largest":
        return max(models, key=lambda x: x['file_size'])['name']
    elif criteria == "smallest":
        return min(models, key=lambda x: x['file_size'])['name']
    else:
        return models[0]['name']  # Default to first model


def validate_all_models(models_dir: str = "models") -> Dict[str, Dict[str, Any]]:
    """
    Validate all models in the specified directory.
    
    Args:
        models_dir: Directory containing models
        
    Returns:
        Dictionary mapping model names to validation results
    """
    manager = ModelManager(models_dir)
    validator = ModelValidator()
    results = {}
    
    for model_file in Path(models_dir).glob("*.zip"):
        model_name = model_file.stem
        validation_result = validator.validate_model_file(model_file)
        results[model_name] = validation_result
    
    return results


def create_model_metadata(model_name: str, algorithm: str = "PPO",
                         hyperparameters: Optional[Dict[str, Any]] = None,
                         notes: str = "") -> ModelMetadata:
    """
    Create ModelMetadata instance with provided information.
    
    Args:
        model_name: Name of the model
        algorithm: Algorithm used for training
        hyperparameters: Training hyperparameters
        notes: Additional notes about the model
        
    Returns:
        ModelMetadata instance
    """
    metadata = ModelMetadata(model_name)
    metadata.algorithm = algorithm
    if hyperparameters:
        metadata.set_hyperparameters(hyperparameters)
    metadata.notes = notes
    
    return metadata


def safe_load_model(model_path: Union[str, Path], 
                   validate: bool = True) -> Optional[BaseAlgorithm]:
    """
    Safely load a model with optional validation.
    
    Args:
        model_path: Path to the model file
        validate: Whether to validate the model before loading
        
    Returns:
        Loaded model or None if failed
    """
    model_path = Path(model_path)
    
    if validate:
        validator = ModelValidator()
        validation_result = validator.validate_model_file(model_path)
        if not validation_result['valid']:
            logging.error(f"Model validation failed: {validation_result['errors']}")
            return None
    
    try:
        return PPO.load(str(model_path))
    except Exception as e:
        logging.error(f"Failed to load model {model_path}: {e}")
        return None 