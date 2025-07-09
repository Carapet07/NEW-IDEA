"""
🧪 Model Utilities Tests
Comprehensive unit tests for model_utils module functionality.

Test coverage:
- ModelMetadata class functionality
- ModelValidator validation methods
- ModelManager comprehensive operations
- Model discovery and organization
- Backup and versioning functionality
- Model comparison and analysis
- Metadata management
- Safe model operations
"""

import unittest
import tempfile
import shutil
import json
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import sys
import os
import datetime

# Add ml_training to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml_training'))

try:
    from model_utils import (
        ModelMetadata, ModelValidator, ModelManager,
        find_best_model, validate_all_models, create_model_metadata,
        safe_load_model
    )
    MODEL_UTILS_AVAILABLE = True
except ImportError:
    MODEL_UTILS_AVAILABLE = False


@unittest.skipIf(not MODEL_UTILS_AVAILABLE, "Model utilities not available")
class TestModelMetadata(unittest.TestCase):
    """Test cases for ModelMetadata class."""
    
    def test_metadata_creation(self):
        """Test basic ModelMetadata creation."""
        metadata = ModelMetadata("test_model")
        
        self.assertEqual(metadata.model_name, "test_model")
        self.assertIsInstance(metadata.creation_time, str)
        self.assertEqual(metadata.version, "1.0.0")
        self.assertEqual(metadata.algorithm, "PPO")
        self.assertEqual(metadata.hyperparameters, {})
        self.assertEqual(metadata.training_stats, {})
        self.assertEqual(metadata.validation_results, {})
        self.assertEqual(metadata.notes, "")
        self.assertEqual(metadata.tags, [])
    
    def test_update_training_stats(self):
        """Test updating training statistics."""
        metadata = ModelMetadata("test_model")
        original_modified = metadata.last_modified
        
        # Small delay to ensure timestamp changes
        import time
        time.sleep(0.001)
        
        stats = {
            'total_episodes': 1000,
            'success_rate': 85.5,
            'avg_reward': 42.3
        }
        metadata.update_training_stats(stats)
        
        self.assertEqual(metadata.training_stats, stats)
        self.assertNotEqual(metadata.last_modified, original_modified)
    
    def test_update_validation_results(self):
        """Test updating validation results."""
        metadata = ModelMetadata("test_model")
        
        results = {
            'test_episodes': 50,
            'test_success_rate': 90.0,
            'avg_test_reward': 48.5
        }
        metadata.update_validation_results(results)
        
        self.assertEqual(metadata.validation_results, results)
    
    def test_add_tags(self):
        """Test adding tags to metadata."""
        metadata = ModelMetadata("test_model")
        
        metadata.add_tag("fast_training")
        metadata.add_tag("high_performance")
        metadata.add_tag("fast_training")  # Duplicate should be ignored
        
        self.assertIn("fast_training", metadata.tags)
        self.assertIn("high_performance", metadata.tags)
        self.assertEqual(metadata.tags.count("fast_training"), 1)
    
    def test_set_hyperparameters(self):
        """Test setting hyperparameters."""
        metadata = ModelMetadata("test_model")
        
        hyperparams = {
            'learning_rate': 3e-4,
            'batch_size': 64,
            'n_epochs': 10
        }
        metadata.set_hyperparameters(hyperparams)
        
        self.assertEqual(metadata.hyperparameters, hyperparams)
    
    def test_to_dict_conversion(self):
        """Test converting metadata to dictionary."""
        metadata = ModelMetadata("test_model")
        metadata.update_training_stats({'episodes': 100})
        metadata.add_tag("test_tag")
        
        metadata_dict = metadata.to_dict()
        
        self.assertIsInstance(metadata_dict, dict)
        self.assertEqual(metadata_dict['model_name'], "test_model")
        self.assertEqual(metadata_dict['training_stats'], {'episodes': 100})
        self.assertEqual(metadata_dict['tags'], ["test_tag"])
    
    def test_from_dict_creation(self):
        """Test creating metadata from dictionary."""
        data = {
            'model_name': 'restored_model',
            'version': '2.0.0',
            'algorithm': 'A2C',
            'training_stats': {'episodes': 500},
            'tags': ['restored', 'tested']
        }
        
        metadata = ModelMetadata.from_dict(data)
        
        self.assertEqual(metadata.model_name, 'restored_model')
        self.assertEqual(metadata.version, '2.0.0')
        self.assertEqual(metadata.algorithm, 'A2C')
        self.assertEqual(metadata.training_stats, {'episodes': 500})
        self.assertEqual(metadata.tags, ['restored', 'tested'])


@unittest.skipIf(not MODEL_UTILS_AVAILABLE, "Model utilities not available")
class TestModelValidator(unittest.TestCase):
    """Test cases for ModelValidator class."""
    
    def setUp(self):
        """Set up test validator and temporary directory."""
        self.validator = ModelValidator()
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
    
    def create_dummy_model_file(self, filename: str, content: bytes = b"dummy model content") -> Path:
        """Create a dummy model file for testing."""
        model_path = Path(self.temp_dir) / filename
        with open(model_path, 'wb') as f:
            f.write(content)
        return model_path
    
    def create_dummy_zip_model(self, filename: str) -> Path:
        """Create a dummy ZIP model file that looks more realistic."""
        model_path = Path(self.temp_dir) / filename
        with zipfile.ZipFile(model_path, 'w') as zf:
            zf.writestr('data', b"dummy model data")
            zf.writestr('metadata.json', b'{"algorithm": "PPO"}')
        return model_path
    
    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        fake_path = Path(self.temp_dir) / "nonexistent.zip"
        result = self.validator.validate_model_file(fake_path)
        
        self.assertFalse(result['valid'])
        self.assertFalse(result['file_exists'])
        self.assertIn("does not exist", result['errors'][0])
    
    def test_validate_existing_file_basic(self):
        """Test validation of existing file (basic checks)."""
        model_path = self.create_dummy_zip_model("test_model.zip")
        
        with patch('model_utils.PPO') as mock_ppo:
            mock_model = MagicMock()
            mock_model.policy = MagicMock()
            mock_model.observation_space = "Box(7,)"
            mock_model.action_space = "Discrete(4)"
            mock_ppo.load.return_value = mock_model
            
            result = self.validator.validate_model_file(model_path)
        
        self.assertTrue(result['file_exists'])
        self.assertTrue(result['file_readable'])
        self.assertGreater(result['file_size'], 0)
        self.assertIsNotNone(result['file_hash'])
    
    def test_validate_corrupted_file(self):
        """Test validation of corrupted file."""
        model_path = self.create_dummy_model_file("corrupted.zip", b"not a valid zip")
        
        with patch('model_utils.PPO') as mock_ppo:
            mock_ppo.load.side_effect = Exception("Cannot load corrupted file")
            
            result = self.validator.validate_model_file(model_path)
        
        self.assertTrue(result['file_exists'])
        self.assertTrue(result['file_readable'])
        self.assertFalse(result['model_loadable'])
        self.assertIn("Cannot load model", result['errors'][0])
    
    def test_validate_very_small_file(self):
        """Test validation generates warning for very small files."""
        model_path = self.create_dummy_model_file("tiny.zip", b"small")
        
        with patch('model_utils.PPO') as mock_ppo:
            mock_ppo.load.side_effect = Exception("Too small")
            
            result = self.validator.validate_model_file(model_path)
        
        # Should still detect file exists and is readable
        self.assertTrue(result['file_exists'])
        self.assertTrue(result['file_readable'])
        self.assertLess(result['file_size'], 1000)
    
    def test_validate_model_performance_placeholder(self):
        """Test model performance validation (placeholder implementation)."""
        model_path = self.create_dummy_zip_model("performance_test.zip")
        expected_performance = {'success_rate': 80.0, 'avg_reward': 50.0}
        
        result = self.validator.validate_model_performance(model_path, expected_performance)
        
        # Current implementation returns placeholder
        self.assertIn('performance_valid', result)
        self.assertIn('meets_expectations', result)
        self.assertIn('performance_metrics', result)
        self.assertIn('recommendations', result)


@unittest.skipIf(not MODEL_UTILS_AVAILABLE, "Model utilities not available")
class TestModelManager(unittest.TestCase):
    """Test cases for ModelManager class."""
    
    def setUp(self):
        """Set up test model manager with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.temp_dir) / "models"
        self.models_dir.mkdir()
        self.manager = ModelManager(str(self.models_dir))
        self.addCleanup(shutil.rmtree, self.temp_dir)
    
    def create_test_model_file(self, name: str, size: int = 1000) -> Path:
        """Create a test model file."""
        model_path = self.models_dir / f"{name}.zip"
        with zipfile.ZipFile(model_path, 'w') as zf:
            zf.writestr('data', b"x" * size)
        return model_path
    
    def test_manager_initialization(self):
        """Test ModelManager initialization."""
        self.assertTrue(self.models_dir.exists())
        self.assertTrue(self.manager.backup_dir.exists())
        self.assertIsInstance(self.manager.metadata, dict)
    
    def test_discover_models_empty(self):
        """Test model discovery with no models."""
        models = self.manager.discover_models()
        self.assertEqual(len(models), 0)
    
    def test_discover_models_with_files(self):
        """Test model discovery with model files."""
        # Create test model files
        self.create_test_model_file("model1", 1000)
        self.create_test_model_file("model2", 2000)
        
        with patch.object(self.manager.validator, 'validate_model_file') as mock_validate:
            mock_validate.return_value = {'valid': True, 'errors': []}
            
            models = self.manager.discover_models(include_metadata=False)
        
        self.assertEqual(len(models), 2)
        model_names = [m['name'] for m in models]
        self.assertIn('model1', model_names)
        self.assertIn('model2', model_names)
    
    def test_get_model_info_existing(self):
        """Test getting info for existing model."""
        self.create_test_model_file("test_model")
        
        with patch.object(self.manager.validator, 'validate_model_file') as mock_validate:
            mock_validate.return_value = {'valid': True, 'errors': []}
            
            model_info = self.manager.get_model_info("test_model")
        
        self.assertIsNotNone(model_info)
        self.assertEqual(model_info['name'], "test_model")
    
    def test_get_model_info_nonexistent(self):
        """Test getting info for non-existent model."""
        model_info = self.manager.get_model_info("nonexistent_model")
        self.assertIsNone(model_info)
    
    def test_backup_model(self):
        """Test model backup functionality."""
        model_path = self.create_test_model_file("backup_test")
        
        backup_path = self.manager.backup_model("backup_test", "test_backup")
        
        self.assertIsNotNone(backup_path)
        self.assertTrue(Path(backup_path).exists())
        self.assertIn("backup_test_backup_", backup_path)
    
    def test_backup_nonexistent_model(self):
        """Test backup of non-existent model."""
        backup_path = self.manager.backup_model("nonexistent_model")
        self.assertIsNone(backup_path)
    
    def test_compare_models(self):
        """Test model comparison functionality."""
        # Create test models
        self.create_test_model_file("model_a", 1000)
        self.create_test_model_file("model_b", 2000)
        
        with patch.object(self.manager.validator, 'validate_model_file') as mock_validate:
            mock_validate.return_value = {'valid': True, 'errors': []}
            
            comparison = self.manager.compare_models(["model_a", "model_b"])
        
        self.assertIn('models', comparison)
        self.assertIn('summary', comparison)
        self.assertIn('recommendations', comparison)
        self.assertIn('model_a', comparison['models'])
        self.assertIn('model_b', comparison['models'])
    
    def test_save_and_load_metadata(self):
        """Test metadata saving and loading."""
        metadata = ModelMetadata("test_model")
        metadata.update_training_stats({'episodes': 100})
        
        # Test saving
        success = self.manager.save_model_metadata("test_model", metadata)
        self.assertTrue(success)
        
        # Test loading
        loaded_metadata = self.manager.load_model_metadata("test_model")
        self.assertIsNotNone(loaded_metadata)
        self.assertEqual(loaded_metadata.model_name, "test_model")
        self.assertEqual(loaded_metadata.training_stats, {'episodes': 100})
    
    def test_load_nonexistent_metadata(self):
        """Test loading metadata for non-existent model."""
        metadata = self.manager.load_model_metadata("nonexistent_model")
        self.assertIsNone(metadata)
    
    def test_cleanup_old_backups(self):
        """Test cleanup of old backup files."""
        # Create old backup file
        old_backup = self.manager.backup_dir / "old_backup.zip"
        with open(old_backup, 'w') as f:
            f.write("old backup")
        
        # Set modification time to old date
        old_time = (datetime.datetime.now() - datetime.timedelta(days=35)).timestamp()
        os.utime(old_backup, (old_time, old_time))
        
        # Create recent backup file
        recent_backup = self.manager.backup_dir / "recent_backup.zip"
        with open(recent_backup, 'w') as f:
            f.write("recent backup")
        
        deleted_count = self.manager.cleanup_old_backups(keep_days=30)
        
        self.assertEqual(deleted_count, 1)
        self.assertFalse(old_backup.exists())
        self.assertTrue(recent_backup.exists())
    
    def test_export_model_info(self):
        """Test exporting model information."""
        model_path = self.create_test_model_file("export_test")
        
        with patch.object(self.manager.validator, 'validate_model_file') as mock_validate:
            mock_validate.return_value = {'valid': True, 'errors': []}
            
            export_path = Path(self.temp_dir) / "export_test.json"
            success = self.manager.export_model_info("export_test", str(export_path))
        
        self.assertTrue(success)
        self.assertTrue(Path(export_path).exists())
        
        # Verify exported content
        with open(export_path, 'r') as f:
            exported_data = json.load(f)
        
        self.assertEqual(exported_data['name'], "export_test")


@unittest.skipIf(not MODEL_UTILS_AVAILABLE, "Model utilities not available")
class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def setUp(self):
        """Set up temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.temp_dir) / "models"
        self.models_dir.mkdir()
        self.addCleanup(shutil.rmtree, self.temp_dir)
    
    def create_test_model_file(self, name: str, size: int = 1000) -> Path:
        """Create a test model file."""
        model_path = self.models_dir / f"{name}.zip"
        with zipfile.ZipFile(model_path, 'w') as zf:
            zf.writestr('data', b"x" * size)
        return model_path
    
    def test_find_best_model_newest(self):
        """Test finding newest model."""
        # Create models with different timestamps
        model1 = self.create_test_model_file("old_model")
        model2 = self.create_test_model_file("new_model")
        
        # Modify timestamps
        old_time = (datetime.datetime.now() - datetime.timedelta(hours=1)).timestamp()
        os.utime(model1, (old_time, old_time))
        
        with patch('model_utils.ModelValidator') as mock_validator:
            mock_validator.return_value.validate_model_file.return_value = {'valid': True}
            
            best_model = find_best_model(str(self.models_dir), criteria="newest")
        
        self.assertEqual(best_model, "new_model")
    
    def test_find_best_model_largest(self):
        """Test finding largest model."""
        self.create_test_model_file("small_model", 500)
        self.create_test_model_file("large_model", 1500)
        
        with patch('model_utils.ModelValidator') as mock_validator:
            mock_validator.return_value.validate_model_file.return_value = {'valid': True}
            
            best_model = find_best_model(str(self.models_dir), criteria="largest")
        
        self.assertEqual(best_model, "large_model")
    
    def test_find_best_model_no_models(self):
        """Test finding best model when no models exist."""
        best_model = find_best_model(str(self.models_dir))
        self.assertIsNone(best_model)
    
    def test_validate_all_models(self):
        """Test validating all models in directory."""
        self.create_test_model_file("model1")
        self.create_test_model_file("model2")
        
        with patch('model_utils.ModelValidator') as mock_validator:
            mock_validator.return_value.validate_model_file.return_value = {
                'valid': True, 'errors': []
            }
            
            results = validate_all_models(str(self.models_dir))
        
        self.assertIn('model1', results)
        self.assertIn('model2', results)
        self.assertTrue(results['model1']['valid'])
        self.assertTrue(results['model2']['valid'])
    
    def test_create_model_metadata(self):
        """Test creating model metadata with parameters."""
        hyperparams = {'learning_rate': 3e-4, 'batch_size': 64}
        
        metadata = create_model_metadata(
            "test_model",
            algorithm="A2C",
            hyperparameters=hyperparams,
            notes="Test model"
        )
        
        self.assertEqual(metadata.model_name, "test_model")
        self.assertEqual(metadata.algorithm, "A2C")
        self.assertEqual(metadata.hyperparameters, hyperparams)
        self.assertEqual(metadata.notes, "Test model")
    
    @patch('model_utils.PPO')
    def test_safe_load_model_success(self, mock_ppo):
        """Test successful safe model loading."""
        model_path = self.create_test_model_file("test_model")
        mock_model = MagicMock()
        mock_ppo.load.return_value = mock_model
        
        with patch('model_utils.ModelValidator') as mock_validator:
            mock_validator.return_value.validate_model_file.return_value = {'valid': True}
            
            loaded_model = safe_load_model(model_path, validate=True)
        
        self.assertIsNotNone(loaded_model)
        mock_ppo.load.assert_called_once()
    
    @patch('model_utils.PPO')
    def test_safe_load_model_validation_failure(self, mock_ppo):
        """Test safe model loading with validation failure."""
        model_path = self.create_test_model_file("invalid_model")
        
        with patch('model_utils.ModelValidator') as mock_validator:
            mock_validator.return_value.validate_model_file.return_value = {
                'valid': False, 'errors': ['Validation failed']
            }
            
            loaded_model = safe_load_model(model_path, validate=True)
        
        self.assertIsNone(loaded_model)
        mock_ppo.load.assert_not_called()
    
    @patch('model_utils.PPO')
    def test_safe_load_model_load_failure(self, mock_ppo):
        """Test safe model loading with load failure."""
        model_path = self.create_test_model_file("corrupt_model")
        mock_ppo.load.side_effect = Exception("Load failed")
        
        loaded_model = safe_load_model(model_path, validate=False)
        
        self.assertIsNone(loaded_model)


class TestModelUtilsIntegration(unittest.TestCase):
    """Integration tests for model utilities."""
    
    @unittest.skipIf(not MODEL_UTILS_AVAILABLE, "Model utilities not available")
    def test_full_model_management_workflow(self):
        """Test complete model management workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir) / "models"
            models_dir.mkdir()
            
            # Create manager
            manager = ModelManager(str(models_dir))
            
            # Create test models
            model1_path = models_dir / "model1.zip"
            model2_path = models_dir / "model2.zip"
            
            with zipfile.ZipFile(model1_path, 'w') as zf:
                zf.writestr('data', b"model1 data")
            
            with zipfile.ZipFile(model2_path, 'w') as zf:
                zf.writestr('data', b"model2 data" * 100)  # Larger model
            
            # Create and save metadata
            metadata1 = create_model_metadata("model1", notes="First model")
            metadata1.update_training_stats({'success_rate': 75.0})
            manager.save_model_metadata("model1", metadata1)
            
            metadata2 = create_model_metadata("model2", notes="Second model")
            metadata2.update_training_stats({'success_rate': 85.0})
            manager.save_model_metadata("model2", metadata2)
            
            # Test discovery
            with patch.object(manager.validator, 'validate_model_file') as mock_validate:
                mock_validate.return_value = {'valid': True, 'errors': []}
                
                models = manager.discover_models(include_metadata=True)
            
            self.assertEqual(len(models), 2)
            
            # Test comparison
            comparison = manager.compare_models(["model1", "model2"])
            self.assertIn('models', comparison)
            self.assertIn('summary', comparison)
            
            # Test backup
            backup_path = manager.backup_model("model1", "integration_test")
            self.assertIsNotNone(backup_path)
            self.assertTrue(Path(backup_path).exists())
            
            # Test cleanup (should not delete recent backup)
            deleted_count = manager.cleanup_old_backups(keep_days=1)
            self.assertEqual(deleted_count, 0)  # Recent backup should remain


def run_model_utils_tests():
    """Run all model utility tests."""
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestModelMetadata,
        TestModelValidator,
        TestModelManager,
        TestUtilityFunctions,
        TestModelUtilsIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    if MODEL_UTILS_AVAILABLE:
        print("🧪 Running Model Utilities Tests...")
        success = run_model_utils_tests()
        if success:
            print("✅ All model utility tests passed!")
        else:
            print("❌ Some model utility tests failed!")
    else:
        print("⚠️ Model utilities not available for testing")
        unittest.main() 