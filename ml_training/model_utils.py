"""
Model Utilities Module

Comprehensive model management utilities including validation, backup, metadata tracking,
automated organization, and interactive CLI. Provides robust model lifecycle management for AI training.
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
import argparse
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
import numpy as np
import logging


def find_model_file(model_name: str, models_dir: str = "models") -> Optional[Path]:
    """
    Find a model file by searching in common locations.
    
    Args:
        model_name: Name of the model to find
        models_dir: Directory to search for models
        
    Returns:
        Path to the model file if found, None otherwise
    """
    models_path = Path(models_dir)
    
    search_paths = [
        models_path / f"{model_name}.zip",
        models_path / model_name,
        Path(f"{model_name}.zip"),
        Path(model_name),
        Path(".") / f"{model_name}.zip",
        Path(".") / model_name
    ]
    
    for path in search_paths:
        if path.exists():
            return path
    return None


def list_available_models(models_dir: str = "models") -> List[str]:
    """
    List all available saved models.
    
    Args:
        models_dir: Directory to search for models
        
    Returns:
        List of available model names
    """
    models = []
    models_path = Path(models_dir)
    
    # Check current directory
    for file in Path(".").glob("*.zip"):
        models.append(file.stem)
    
    # Check models directory
    if models_path.exists():
        for file in models_path.glob("*.zip"):
            models.append(file.stem)
    
    return sorted(set(models))


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
        """Set hyperparameters for the model."""
        self.hyperparameters = hyperparams
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
        """Create metadata from dictionary."""
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
        
        # Check if file exists
        if not model_path.exists():
            validation_result['errors'].append(f"Model file {model_path} does not exist")
            return validation_result
        
        validation_result['file_exists'] = True
        validation_result['file_size'] = model_path.stat().st_size
        
        # Check if file is readable
        try:
            with open(model_path, 'rb') as f:
                file_content = f.read(1024)  # Read first 1KB
            validation_result['file_readable'] = True
            
            # Calculate file hash for integrity
            with open(model_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            validation_result['file_hash'] = file_hash
            
        except Exception as e:
            validation_result['errors'].append(f"Cannot read model file: {e}")
            return validation_result
        
        # Check if file is too small (likely corrupted)
        if validation_result['file_size'] < 1000:  # Less than 1KB
            validation_result['warnings'].append("Model file is very small, might be corrupted")
        
        # Try to load model (basic check)
        try:
            # For now, just check if it's a valid zip file (SB3 format)
            if model_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(model_path, 'r') as zf:
                    file_list = zf.namelist()
                    validation_result['model_info']['zip_contents'] = file_list
                    validation_result['model_loadable'] = True
            else:
                validation_result['warnings'].append("Model file is not in .zip format")
                
        except Exception as e:
            validation_result['errors'].append(f"Cannot load model: {e}")
            return validation_result
        
        # If we get here, basic validation passed
        if not validation_result['errors']:
            validation_result['valid'] = True
        
        return validation_result
    
    def validate_model_performance(self, model_path: Union[str, Path], 
                                 expected_performance: Dict[str, float]) -> Dict[str, Any]:
        """
        Validate model performance against expected metrics.
        
        Args:
            model_path: Path to the model file
            expected_performance: Dictionary of expected performance metrics
            
        Returns:
            Dictionary containing performance validation results
        """
        # This is a placeholder for performance validation
        # Would require running the model in an environment
        return {
            'performance_valid': True,
            'performance_metrics': {},
            'warnings': ["Performance validation not implemented yet"]
        }


class ModelManager:
    """
    Comprehensive model management utility with interactive capabilities.
    
    Provides functionality for:
    - Model discovery and listing
    - Model backup and versioning
    - Model comparison and selection
    - Model cleanup and organization
    - Interactive CLI for user-friendly management
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
        
        # Legacy metadata file for compatibility
        self.legacy_metadata_file = self.models_dir / ".model_metadata.json"
        self._load_legacy_metadata()
    
    def _load_legacy_metadata(self) -> None:
        """Load legacy metadata from single file for backward compatibility."""
        try:
            if self.legacy_metadata_file.exists():
                with open(self.legacy_metadata_file, 'r') as f:
                    self.metadata = json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load legacy metadata: {e}")
            self.metadata = {}
    
    def _save_legacy_metadata(self) -> None:
        """Save legacy metadata to single file for backward compatibility."""
        try:
            with open(self.legacy_metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error saving legacy metadata: {e}")

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
                # Try to load from individual metadata file first
                metadata_file = self.models_dir / f"{model_file.stem}_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata_dict = json.load(f)
                        model_info['metadata'] = ModelMetadata.from_dict(metadata_dict)
                    except Exception as e:
                        self.logger.warning(f"Failed to load metadata for {model_file.stem}: {e}")
                
                # Fall back to legacy metadata
                elif model_file.stem in self.metadata:
                    legacy_data = self.metadata[model_file.stem]
                    model_info['legacy_metadata'] = legacy_data
            
            models.append(model_info)
        
        # Sort by modification time (newest first)
        models.sort(key=lambda x: x['modified_time'], reverse=True)
        
        return models
    
    def list_models(self, detailed: bool = False, sort_by: str = "name") -> None:
        """
        List all available models with optional detailed information.
        
        Args:
            detailed: Whether to show detailed information
            sort_by: Sort criteria ("name", "size", "date", "performance")
        """
        models = []
        
        # Scan for model files
        for file_path in self.models_dir.glob("*.zip"):
            if file_path.name.startswith('.'):
                continue
                
            model_name = file_path.stem
            file_size = file_path.stat().st_size / 1024  # KB
            modified_time = datetime.datetime.fromtimestamp(file_path.stat().st_mtime)
            
            model_info = {
                'name': model_name,
                'path': file_path,
                'size_kb': file_size,
                'modified': modified_time,
                'metadata': self.metadata.get(model_name, {}),
                'legacy_metadata': self.metadata.get(model_name, {})
            }
            models.append(model_info)
        
        if not models:
            print("No trained models found.")
            print("Train a model first using: python escape_cage_trainer.py")
            return
        
        # Sort models
        sort_key_map = {
            "name": lambda x: x['name'],
            "size": lambda x: x['size_kb'],
            "date": lambda x: x['modified'],
            "performance": lambda x: x['legacy_metadata'].get('success_rate', 0)
        }
        
        if sort_by in sort_key_map:
            models.sort(key=sort_key_map[sort_by], reverse=(sort_by != "name"))
        
        # Display models
        print("AI Model Collection")
        print("=" * 50)
        
        if detailed:
            for model in models:
                print(f"\nModel: {model['name']}")
                print(f"  Size: {model['size_kb']:.1f} KB")
                print(f"  Modified: {model['modified'].strftime('%Y-%m-%d %H:%M')}")
                
                # Show performance if available
                meta = model['legacy_metadata']
                if 'success_rate' in meta:
                    print(f"  Success Rate: {meta['success_rate']}%")
                if 'total_steps' in meta:
                    print(f"  Training Steps: {meta['total_steps']:,}")
                if 'notes' in meta:
                    print(f"  Notes: {meta['notes']}")
        else:
            print(f"{'Name':<25} {'Size':<10} {'Modified':<16} {'Success%':<8}")
            print("-" * 65)
            
            for model in models:
                meta = model['legacy_metadata']
                success_rate = meta.get('success_rate', 'N/A')
                success_str = f"{success_rate}%" if success_rate != 'N/A' else 'N/A'
                
                print(f"{model['name']:<25} {model['size_kb']:<9.1f}K "
                      f"{model['modified'].strftime('%m/%d %H:%M'):<16} {success_str:<8}")
        
        print(f"\nTotal models: {len(models)}")

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

    def organize_models(self) -> None:
        """Organize loose model files into the models directory."""
        print("Organizing models...")
        
        # Look for .zip files in current directory
        current_dir = Path(".")
        model_files = list(current_dir.glob("*.zip"))
        
        # Filter out files already in models directory
        files_to_move = [f for f in model_files if f.parent != self.models_dir]
        
        if not files_to_move:
            print("All models already organized")
            return
        
        print("Found model files to organize:")
        for file in files_to_move:
            print(f"  {file.name}")
        
        confirm = input(f"\nMove {len(files_to_move)} files to models/? (y/N): ").strip().lower()
        if confirm == 'y':
            moved_count = 0
            for file in files_to_move:
                try:
                    destination = self.models_dir / file.name
                    shutil.move(str(file), str(destination))
                    print(f"Moved {file.name} → models/")
                    moved_count += 1
                except Exception as e:
                    print(f"Failed to move {file.name}: {e}")
            
            print(f"Organized {moved_count} models into models/ directory")
        else:
            print("Organization cancelled")

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
            print(f"Model '{model_name}' not found!")
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
            
            # Update legacy metadata with backup info
            if model_name not in self.metadata:
                self.metadata[model_name] = {}
            if 'backups' not in self.metadata[model_name]:
                self.metadata[model_name]['backups'] = []
            
            file_size = backup_path.stat().st_size / (1024 * 1024)  # MB
            self.metadata[model_name]['backups'].append({
                'timestamp': timestamp,
                'path': str(backup_path),
                'size_mb': file_size
            })
            
            self._save_legacy_metadata()
            
            print(f"Backup created: {backup_name} ({file_size:.1f} MB)")
            self.logger.info(f"Model {model_name} backed up to {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            print(f"Backup failed: {e}")
            self.logger.error(f"Failed to backup model {model_name}: {e}")
            return None

    def delete_model(self, model_name: str, force: bool = False) -> bool:
        """
        Delete a model with safety checks.
        
        Args:
            model_name: Name of the model to delete
            force: Skip safety confirmations
            
        Returns:
            True if deletion was successful, False otherwise
        """
        model_path = self.models_dir / f"{model_name}.zip"
        
        if not model_path.exists():
            print(f"Model '{model_name}' not found!")
            return False
        
        if not force:
            # Safety confirmation
            print(f"\nWARNING: You are about to delete '{model_name}'")
            print("This action cannot be undone!")
            
            confirm = input("Type 'DELETE' to confirm: ").strip()
            if confirm != 'DELETE':
                print("Deletion cancelled")
                return False
            
            # Offer backup
            create_backup = input("Create backup before deletion? (Y/n): ").strip().lower()
            if create_backup != 'n':
                if not self.backup_model(model_name):
                    print("Backup failed. Aborting deletion for safety.")
                    return False
        
        # Confirm deletion one more time
        final_confirm = input(f"Final confirmation - delete '{model_name}'? (type model name): ").strip()
        if final_confirm != model_name:
            print("Confirmation text doesn't match. Deletion cancelled.")
            return False
        
        try:
            # Delete the model file
            model_path.unlink()
            
            # Delete individual metadata file if exists
            metadata_path = self.models_dir / f"{model_name}_metadata.json"
            if metadata_path.exists():
                metadata_path.unlink()
            
            # Remove from legacy metadata
            if model_name in self.metadata:
                del self.metadata[model_name]
                self._save_legacy_metadata()
            
            print(f"Successfully deleted '{model_name}'")
            return True
            
        except Exception as e:
            print(f"Deletion failed: {e}")
            return False

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

    def compare_two_models(self, model1: str, model2: str) -> None:
        """
        Compare two models side by side with detailed output.
        
        Args:
            model1: Name of first model
            model2: Name of second model
        """
        print(f"Comparing '{model1}' vs '{model2}'")
        print("=" * 50)
        
        # Check if models exist
        path1 = self.models_dir / f"{model1}.zip"
        path2 = self.models_dir / f"{model2}.zip"
        
        if not path1.exists():
            print(f"Model '{model1}' not found!")
            return
        if not path2.exists():
            print(f"Model '{model2}' not found!")
            return
        
        # Get file information
        size1 = path1.stat().st_size / 1024  # KB
        size2 = path2.stat().st_size / 1024  # KB
        
        date1 = datetime.datetime.fromtimestamp(path1.stat().st_mtime)
        date2 = datetime.datetime.fromtimestamp(path2.stat().st_mtime)
        
        # Get metadata
        meta1 = self.metadata.get(model1, {})
        meta2 = self.metadata.get(model2, {})
        
        # Display comparison
        print(f"{'Metric':<15} | {model1:<20} | {model2:<20}")
        print("-" * 60)
        print(f"{'Size':<15} | {size1:<18.1f} KB | {size2:<18.1f} KB")
        print(f"{'Modified':<15} | {date1.strftime('%Y-%m-%d %H:%M'):<20} | {date2.strftime('%Y-%m-%d %H:%M'):<20}")
        
        # Performance comparison
        rate1 = meta1.get('success_rate', 'Unknown')
        rate2 = meta2.get('success_rate', 'Unknown')
        print(f"{'Success Rate':<15} | {str(rate1):<18}%  | {str(rate2):<18}%")
        
        steps1 = meta1.get('total_steps', 'Unknown')
        steps2 = meta2.get('total_steps', 'Unknown')
        print(f"{'Steps':<15} | {str(steps1):<20} | {str(steps2):<20}")
        
        print(f"\nRECOMMENDATIONS:")
        
        # Size recommendation
        if isinstance(size1, (int, float)) and isinstance(size2, (int, float)):
            if size1 > size2 * 1.5:
                print(f"  - {model1} is significantly larger - may have more complex learned behavior")
            elif size2 > size1 * 1.5:
                print(f"  - {model2} is significantly larger - may have more complex learned behavior")
        
        # Performance recommendation
        if isinstance(rate1, (int, float)) and isinstance(rate2, (int, float)):
            if rate1 > rate2:
                print(f"  - {model1} has better success rate ({rate1}% vs {rate2}%)")
            elif rate2 > rate1:
                print(f"  - {model2} has better success rate ({rate2}% vs {rate1}%)")
            else:
                print("  - Both models have similar performance")
        
        # Date recommendation
        if date1 > date2:
            print(f"  - {model1} is newer - may have benefited from improved training")
        elif date2 > date1:
            print(f"  - {model2} is newer - may have benefited from improved training")

    def add_model_metadata(self, model_name: str, **metadata_fields) -> bool:
        """
        Add metadata to a model.
        
        Args:
            model_name: Name of the model
            **metadata_fields: Metadata fields to add
            
        Returns:
            True if metadata was added successfully
        """
        if model_name not in self.metadata:
            self.metadata[model_name] = {}
        
        # Add timestamp
        self.metadata[model_name]['last_updated'] = datetime.datetime.now().isoformat()
        
        # Add provided fields
        for key, value in metadata_fields.items():
            self.metadata[model_name][key] = value
        
        self._save_legacy_metadata()
        print(f"Metadata added to '{model_name}'")
        return True

    def cleanup_old_models(self, keep_days: int = 30, dry_run: bool = False) -> None:
        """
        Clean up models older than specified days.
        
        Args:
            keep_days: Number of days to keep models
            dry_run: If True, only show what would be deleted
        """
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=keep_days)
        old_models = []
        
        for model_path in self.models_dir.glob("*.zip"):
            if model_path.name.startswith('.'):
                continue
                
            modified_time = datetime.datetime.fromtimestamp(model_path.stat().st_mtime)
            if modified_time < cutoff_date:
                old_models.append((model_path, modified_time))
        
        if not old_models:
            print(f"No models older than {keep_days} days found")
            return
        
        print(f"Found {len(old_models)} models older than {keep_days} days:")
        for model_path, modified_time in old_models:
            print(f"  - {model_path.name} ({modified_time.strftime('%Y-%m-%d')})")
        
        if dry_run:
            print("(Dry run - no files were deleted)")
            return
        
        confirm = input(f"\nDelete {len(old_models)} old models? (y/N): ").strip().lower()
        if confirm == 'y':
            cleaned_count = 0
            for model_path, _ in old_models:
                try:
                    name = model_path.stem
                    model_path.unlink()
                    
                    # Remove metadata files
                    metadata_path = self.models_dir / f"{name}_metadata.json"
                    if metadata_path.exists():
                        metadata_path.unlink()
                    
                    # Remove from legacy metadata
                    if name in self.metadata:
                        del self.metadata[name]
                    
                    print(f"Removed: {name}")
                    cleaned_count += 1
                except Exception as e:
                    print(f"Failed to remove {name}: {e}")
            
            self._save_legacy_metadata()
            print(f"Cleaned {cleaned_count} old models")
        else:
            print("Cleanup cancelled")

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
    Create model metadata with initial parameters.
    
    Args:
        model_name: Name of the model
        algorithm: Algorithm used for training
        hyperparameters: Training hyperparameters
        notes: Additional notes
        
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
    Safely load a model with validation.
    
    Args:
        model_path: Path to the model file
        validate: Whether to validate before loading
        
    Returns:
        Loaded model or None if loading failed
    """
    model_path = Path(model_path)
    
    if validate:
        validator = ModelValidator()
        validation_result = validator.validate_model_file(model_path)
        if not validation_result['valid']:
            return None
    
    try:
        model = PPO.load(str(model_path))
        return model
    except Exception:
        return None


def main():
    """Main function for simplified command-line interface."""
    parser = argparse.ArgumentParser(description="AI Model Manager - Simple Model Management")
    parser.add_argument("action", choices=["list", "backup", "delete", "compare"],
                        help="Action to perform")
    parser.add_argument("--model", help="Model name for single-model operations")
    parser.add_argument("--model2", help="Second model name for comparisons")
    
    args = parser.parse_args()
    manager = ModelManager()
    
    try:
        if args.action == "list":
            # Always use simple non-detailed listing, sorted by name
            manager.list_models(detailed=False, sort_by="name")
        
        elif args.action == "backup":
            if not args.model:
                print("--model required for backup")
                print("Example: python ml_training/model_utils.py backup --model my_model")
                return
            manager.backup_model(args.model)
        
        elif args.action == "delete":
            if not args.model:
                print("--model required for delete")
                print("Example: python ml_training/model_utils.py delete --model my_model")
                return
            manager.delete_model(args.model, force=False)
        
        elif args.action == "compare":
            if not args.model or not args.model2:
                print("--model and --model2 required for compare")
                print("Example: python ml_training/model_utils.py compare --model model1 --model2 model2")
                return
            manager.compare_two_models(args.model, args.model2)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main() 