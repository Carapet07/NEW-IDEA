"""
📦 AI Model Manager
Enhanced model organization, comparison, and management with safety features
"""

import os
import shutil
import datetime
from pathlib import Path
import json
from typing import List, Dict, Any, Optional, Tuple
import sys


class ModelManager:
    """
    Enhanced AI model manager with comprehensive features for organization, 
    comparison, and safe management of trained models.
    
    Features:
    - Safe deletion with multiple confirmation levels
    - Automatic backup creation before dangerous operations
    - Enhanced metadata tracking and reporting
    - Interactive model selection
    - Comprehensive model analysis and comparison
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.metadata_file = self.models_dir / "model_metadata.json"
        self.metadata = self.load_metadata()
        
        # Create backup directory
        self.backup_dir = self.models_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
    
    def load_metadata(self) -> Dict[str, Any]:
        """Load model metadata from file with error handling."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"⚠️ Warning: Could not load metadata: {e}")
            print("💡 Creating new metadata file...")
        return {}
    
    def save_metadata(self) -> bool:
        """Save model metadata to file with error handling."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"❌ Error saving metadata: {e}")
            return False
    
    def list_models(self, detailed: bool = False, sort_by: str = "modified") -> List[str]:
        """
        List all available models with enhanced details and sorting options.
        
        Args:
            detailed: Whether to show detailed information
            sort_by: Sort criteria ('name', 'size', 'modified', 'success_rate')
            
        Returns:
            List of model names
        """
        print("📦 AI Model Collection")
        print("=" * 70)
        
        # Collect all model files
        model_files = []
        
        # Current directory models
        for file in Path('.').glob('*.zip'):
            if file.name != 'model_metadata.json':
                model_files.append(('current', file))
        
        # Models directory
        for file in self.models_dir.glob('*.zip'):
            model_files.append(('models', file))
        
        if not model_files:
            print("❌ No trained models found.")
            print("🏋️ Train a model first using: python escape_cage_trainer.py")
            return []
        
        # Sort models
        model_files = self._sort_models(model_files, sort_by)
        
        model_names = []
        for i, (location, model_path) in enumerate(model_files, 1):
            model_name = model_path.stem
            model_names.append(model_name)
            
            if detailed:
                self._print_detailed_model_info(i, model_name, location, model_path)
            else:
                self._print_basic_model_info(i, model_name, location, model_path)
        
        print(f"\nTotal models: {len(model_files)}")
        return model_names
    
    def _sort_models(self, model_files: List[Tuple[str, Path]], sort_by: str) -> List[Tuple[str, Path]]:
        """Sort model files by specified criteria."""
        if sort_by == "name":
            return sorted(model_files, key=lambda x: x[1].stem.lower())
        elif sort_by == "size":
            return sorted(model_files, key=lambda x: x[1].stat().st_size, reverse=True)
        elif sort_by == "modified":
            return sorted(model_files, key=lambda x: x[1].stat().st_mtime, reverse=True)
        elif sort_by == "success_rate":
            def get_success_rate(model_file):
                model_name = model_file[1].stem
                meta = self.metadata.get(model_name, {})
                rate_str = meta.get('success_rate', '0%')
                return float(rate_str.replace('%', '')) if rate_str != 'Unknown' else 0
            return sorted(model_files, key=get_success_rate, reverse=True)
        else:
            return model_files
    
    def _print_detailed_model_info(self, index: int, model_name: str, location: str, model_path: Path):
        """Print detailed information about a model."""
        file_size = model_path.stat().st_size / 1024  # KB
        modified_time = datetime.datetime.fromtimestamp(model_path.stat().st_mtime)
        
        print(f"\n{index:2d}. 📋 {model_name}")
        print(f"    📍 Location: {location}")
        print(f"    📏 Size: {file_size:.1f} KB")
        print(f"    📅 Modified: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Show metadata if available
        if model_name in self.metadata:
            meta = self.metadata[model_name]
            print(f"    🎯 Training Steps: {meta.get('total_steps', 'Unknown')}")
            print(f"    📊 Success Rate: {meta.get('success_rate', 'Unknown')}")
            print(f"    📝 Notes: {meta.get('notes', 'None')}")
            if 'backup_date' in meta:
                print(f"    💾 Backup: {meta.get('original_model', 'Unknown')}")
    
    def _print_basic_model_info(self, index: int, model_name: str, location: str, model_path: Path):
        """Print basic information about a model."""
        file_size = model_path.stat().st_size / 1024  # KB
        modified_time = datetime.datetime.fromtimestamp(model_path.stat().st_mtime)
        
        # Get success rate for display
        success_rate = self.metadata.get(model_name, {}).get('success_rate', 'Unknown')
        
        print(f"{index:2d}. 📦 {model_name:<25} | {file_size:>6.1f} KB | "
              f"{modified_time.strftime('%Y-%m-%d %H:%M')} | {success_rate:>8s}")
    
    def interactive_model_selection(self, prompt: str = "Select a model", 
                                  allow_cancel: bool = True) -> Optional[str]:
        """
        Interactive model selection with user-friendly interface.
        
        Args:
            prompt: Prompt to display to user
            allow_cancel: Whether to allow cancellation
            
        Returns:
            Selected model name or None if cancelled
        """
        models = self.list_models(detailed=False)
        
        if not models:
            return None
        
        print(f"\n{prompt}:")
        
        while True:
            try:
                if allow_cancel:
                    choice = input(f"Enter model number (1-{len(models)}) or 'c' to cancel: ").strip().lower()
                    if choice == 'c':
                        print("❌ Operation cancelled")
                        return None
                else:
                    choice = input(f"Enter model number (1-{len(models)}): ").strip()
                
                model_index = int(choice) - 1
                if 0 <= model_index < len(models):
                    selected_model = models[model_index]
                    print(f"✅ Selected: {selected_model}")
                    return selected_model
                else:
                    print(f"❌ Please enter a number between 1 and {len(models)}")
                    
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n❌ Operation cancelled")
                return None
    
    def organize_models(self):
        """Move all models to organized directory structure with confirmation."""
        print("🗂️ Organizing models...")
        
        # Find models to move
        models_to_move = []
        for file in Path('.').glob('*.zip'):
            if file.name != 'model_metadata.json':
                dest = self.models_dir / file.name
                if not dest.exists():
                    models_to_move.append(file)
        
        if not models_to_move:
            print("✅ All models already organized")
            return
        
        print(f"Found {len(models_to_move)} models to organize:")
        for file in models_to_move:
            print(f"  📁 {file.name}")
        
        # Confirm organization
        if self._confirm_action("Organize these models"):
            moved_count = 0
            for file in models_to_move:
                try:
                    dest = self.models_dir / file.name
                    shutil.move(str(file), str(dest))
                    print(f"📁 Moved {file.name} → models/")
                    moved_count += 1
                except Exception as e:
                    print(f"❌ Failed to move {file.name}: {e}")
            
            print(f"✅ Organized {moved_count} models into models/ directory")
        else:
            print("❌ Organization cancelled")
    
    def backup_model(self, model_name: str, notes: str = "", 
                    interactive: bool = False) -> bool:
        """
        Create a timestamped backup of a model with enhanced features.
        
        Args:
            model_name: Name of model to backup
            notes: Notes about the backup
            interactive: Whether to use interactive mode
            
        Returns:
            True if backup was successful
        """
        if interactive:
            model_name = self.interactive_model_selection("Select model to backup")
            if not model_name:
                return False
            
            notes = input("Enter backup notes (optional): ").strip()
        
        model_path = self.find_model(model_name)
        if not model_path:
            print(f"❌ Model '{model_name}' not found!")
            return False
        
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{model_name}_backup_{timestamp}"
            backup_path = self.backup_dir / f"{backup_name}.zip"
            
            # Create backup with progress indication
            print(f"💾 Creating backup of '{model_name}'...")
            file_size = model_path.stat().st_size / (1024 * 1024)  # MB
            
            shutil.copy2(str(model_path), str(backup_path))
            
            # Save metadata
            self.metadata[backup_name] = {
                'original_model': model_name,
                'backup_date': datetime.datetime.now(),
                'notes': notes,
                'backup': True,
                'original_size_mb': file_size
            }
            
            if self.save_metadata():
                print(f"✅ Backup created: {backup_name} ({file_size:.1f} MB)")
                if notes:
                    print(f"📝 Notes: {notes}")
                return True
            else:
                print("⚠️ Backup created but metadata save failed")
                return False
                
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return False
    
    def find_model(self, model_name: str) -> Optional[Path]:
        """Find a model file by name with enhanced search."""
        # Search locations in order of preference
        search_locations = [
            self.models_dir / f"{model_name}.zip",
            Path(f"{model_name}.zip"),
            self.models_dir / model_name,
            Path(model_name),
            self.backup_dir / f"{model_name}.zip"
        ]
        
        for path in search_locations:
            if path.exists():
                return path
        
        return None
    
    def safe_delete_model(self, model_name: str, force: bool = False, 
                         interactive: bool = False) -> bool:
        """
        Safely delete a model with multiple confirmation levels and backup option.
        
        Args:
            model_name: Name of model to delete
            force: Skip all confirmations
            interactive: Use interactive mode
            
        Returns:
            True if deletion was successful
        """
        if interactive:
            model_name = self.interactive_model_selection("Select model to delete")
            if not model_name:
                return False
        
        model_path = self.find_model(model_name)
        if not model_path:
            print(f"❌ Model '{model_name}' not found!")
            return False
        
        # Display model information
        self._display_model_details(model_name, model_path)
        
        if not force:
            # Multi-level confirmation for safety
            print(f"\n⚠️ WARNING: You are about to delete '{model_name}'")
            print("This action cannot be undone!")
            
            # First confirmation
            if not self._confirm_action("Continue with deletion"):
                print("❌ Deletion cancelled")
                return False
            
            # Offer backup option
            create_backup = input("🛡️ Create backup before deletion? (Y/n): ").strip().lower()
            if create_backup != 'n':
                print("💾 Creating safety backup...")
                if not self.backup_model(model_name, "Pre-deletion safety backup"):
                    print("❌ Backup failed. Aborting deletion for safety.")
                    return False
            
            # Final confirmation
            confirmation_text = f"DELETE {model_name}"
            user_input = input(f"🔥 Type '{confirmation_text}' to confirm deletion: ").strip()
            
            if user_input != confirmation_text:
                print("❌ Confirmation text doesn't match. Deletion cancelled.")
                return False
        
        # Perform deletion
        try:
            model_path.unlink()
            
            # Remove from metadata
            if model_name in self.metadata:
                del self.metadata[model_name]
                self.save_metadata()
            
            print(f"🗑️ Successfully deleted '{model_name}'")
            return True
            
        except Exception as e:
            print(f"❌ Deletion failed: {e}")
            return False
    
    def _display_model_details(self, model_name: str, model_path: Path):
        """Display detailed information about a model before operations."""
        file_size = model_path.stat().st_size / 1024  # KB
        modified_time = datetime.datetime.fromtimestamp(model_path.stat().st_mtime)
        
        print(f"\n📋 Model Details:")
        print(f"   Name: {model_name}")
        print(f"   Path: {model_path}")
        print(f"   Size: {file_size:.1f} KB")
        print(f"   Modified: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Show metadata if available
        if model_name in self.metadata:
            meta = self.metadata[model_name]
            print(f"   Training Steps: {meta.get('total_steps', 'Unknown')}")
            print(f"   Success Rate: {meta.get('success_rate', 'Unknown')}")
            if meta.get('notes'):
                print(f"   Notes: {meta.get('notes')}")
    
    def _confirm_action(self, action: str) -> bool:
        """Get user confirmation for an action."""
        try:
            response = input(f"❓ {action}? (y/N): ").strip().lower()
            return response in ['y', 'yes']
        except KeyboardInterrupt:
            print("\n❌ Operation cancelled")
            return False
    
    def add_model_info(self, model_name: str, success_rate: Optional[float] = None, 
                      total_steps: Optional[int] = None, notes: str = "", 
                      interactive: bool = False) -> bool:
        """
        Add metadata information to a model with interactive input.
        
        Args:
            model_name: Name of the model
            success_rate: Success rate percentage
            total_steps: Total training steps
            notes: Additional notes
            interactive: Whether to use interactive input
            
        Returns:
            True if metadata was successfully updated
        """
        if interactive:
            model_name = self.interactive_model_selection("Select model to update")
            if not model_name:
                return False
            
            # Interactive metadata input
            try:
                steps_input = input("Enter total training steps (optional): ").strip()
                if steps_input:
                    total_steps = int(steps_input)
                
                rate_input = input("Enter success rate % (optional): ").strip()
                if rate_input:
                    success_rate = float(rate_input)
                
                notes = input("Enter notes (optional): ").strip()
                
            except ValueError as e:
                print(f"❌ Invalid input: {e}")
                return False
        
        if not self.find_model(model_name):
            print(f"❌ Model '{model_name}' not found!")
            return False
        
        # Initialize metadata entry if it doesn't exist
        if model_name not in self.metadata:
            self.metadata[model_name] = {}
        
        # Update metadata
        updated_fields = []
        if success_rate is not None:
            self.metadata[model_name]['success_rate'] = f"{success_rate:.1f}%"
            updated_fields.append(f"success rate: {success_rate:.1f}%")
        
        if total_steps is not None:
            self.metadata[model_name]['total_steps'] = total_steps
            updated_fields.append(f"training steps: {total_steps:,}")
        
        if notes:
            self.metadata[model_name]['notes'] = notes
            updated_fields.append("notes")
        
        self.metadata[model_name]['last_updated'] = datetime.datetime.now()
        
        if self.save_metadata():
            print(f"📝 Updated metadata for '{model_name}'")
            if updated_fields:
                print(f"   Updated: {', '.join(updated_fields)}")
            return True
        else:
            print(f"❌ Failed to save metadata for '{model_name}'")
            return False
    
    def compare_models(self, model1: str, model2: str, detailed: bool = True):
        """
        Compare two models side by side with enhanced analysis.
        
        Args:
            model1: Name of first model
            model2: Name of second model
            detailed: Whether to show detailed comparison
        """
        print(f"⚖️ Comparing '{model1}' vs '{model2}'")
        print("=" * 80)
        
        path1 = self.find_model(model1)
        path2 = self.find_model(model2)
        
        if not path1:
            print(f"❌ Model '{model1}' not found!")
            return
        if not path2:
            print(f"❌ Model '{model2}' not found!")
            return
        
        # File information
        size1 = path1.stat().st_size / 1024
        size2 = path2.stat().st_size / 1024
        time1 = datetime.datetime.fromtimestamp(path1.stat().st_mtime)
        time2 = datetime.datetime.fromtimestamp(path2.stat().st_mtime)
        
        # Basic comparison
        print(f"\n{'Metric':<20} {'Model 1':<25} {'Model 2':<25}")
        print(f"{'='*20} {'='*25} {'='*25}")
        print(f"{'Name':<20} {model1:<25} {model2:<25}")
        print(f"{'Size':<20} {size1:.1f} KB{'':<19} {size2:.1f} KB")
        print(f"{'Modified':<20} {time1.strftime('%Y-%m-%d %H:%M'):<25} {time2.strftime('%Y-%m-%d %H:%M')}")
        
        # Metadata comparison
        meta1 = self.metadata.get(model1, {})
        meta2 = self.metadata.get(model2, {})
        
        print(f"{'Training Steps':<20} {str(meta1.get('total_steps', 'Unknown')):<25} {str(meta2.get('total_steps', 'Unknown'))}")
        print(f"{'Success Rate':<20} {meta1.get('success_rate', 'Unknown'):<25} {meta2.get('success_rate', 'Unknown')}")
        print(f"{'Notes':<20} {meta1.get('notes', 'None'):<25} {meta2.get('notes', 'None')}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        self._provide_model_recommendations(model1, model2, meta1, meta2, size1, size2, time1, time2)
    
    def _provide_model_recommendations(self, model1: str, model2: str, 
                                     meta1: Dict, meta2: Dict, 
                                     size1: float, size2: float,
                                     time1: datetime.datetime, time2: datetime.datetime):
        """Provide intelligent recommendations based on model comparison."""
        # Compare success rates
        rate1_str = meta1.get('success_rate', '0%')
        rate2_str = meta2.get('success_rate', '0%')
        
        try:
            rate1 = float(rate1_str.replace('%', '')) if rate1_str != 'Unknown' else None
            rate2 = float(rate2_str.replace('%', '')) if rate2_str != 'Unknown' else None
            
            if rate1 is not None and rate2 is not None:
                if rate1 > rate2:
                    print(f"   🏆 {model1} has better performance ({rate1}% vs {rate2}%)")
                elif rate2 > rate1:
                    print(f"   🏆 {model2} has better performance ({rate2}% vs {rate1}%)")
                else:
                    print(f"   ⚖️ Both models have equal performance ({rate1}%)")
            
            # Compare training steps
            steps1 = meta1.get('total_steps')
            steps2 = meta2.get('total_steps')
            
            if steps1 and steps2:
                if steps1 > steps2:
                    print(f"   📚 {model1} has more training ({steps1:,} vs {steps2:,} steps)")
                elif steps2 > steps1:
                    print(f"   📚 {model2} has more training ({steps2:,} vs {steps1:,} steps)")
            
            # Age comparison
            if time1 > time2:
                print(f"   🕐 {model1} is newer (modified {time1.strftime('%Y-%m-%d')})")
            elif time2 > time1:
                print(f"   🕐 {model2} is newer (modified {time2.strftime('%Y-%m-%d')})")
                
        except ValueError:
            print("   ⚠️ Could not compare performance metrics")
    
    def interactive_cleanup(self, keep_days: int = 30):
        """
        Interactive cleanup of old models with user selection.
        
        Args:
            keep_days: Number of days to keep models
        """
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=keep_days)
        
        # Find candidates for cleanup
        cleanup_candidates = []
        for model_path in self.models_dir.glob('*.zip'):
            model_name = model_path.stem
            
            # Only consider backup and partial models
            if 'backup' in model_name or 'partial' in model_name:
                modified_time = datetime.datetime.fromtimestamp(model_path.stat().st_mtime)
                if modified_time < cutoff_date:
                    cleanup_candidates.append((model_name, model_path, modified_time))
        
        if not cleanup_candidates:
            print(f"✅ No models older than {keep_days} days found")
            return
        
        print(f"🧹 Found {len(cleanup_candidates)} models older than {keep_days} days:")
        for i, (name, path, modified) in enumerate(cleanup_candidates, 1):
            age_days = (datetime.datetime.now() - modified).days
            size_kb = path.stat().st_size / 1024
            print(f"   {i:2d}. {name} ({age_days} days old, {size_kb:.1f} KB)")
        
        if self._confirm_action(f"Delete these {len(cleanup_candidates)} old models"):
            cleaned_count = 0
            for name, path, _ in cleanup_candidates:
                try:
                    path.unlink()
                    if name in self.metadata:
                        del self.metadata[name]
                    print(f"🗑️ Removed: {name}")
                    cleaned_count += 1
                except Exception as e:
                    print(f"❌ Failed to remove {name}: {e}")
            
            if cleaned_count > 0:
                self.save_metadata()
                print(f"✅ Cleaned {cleaned_count} old models")
        else:
            print("❌ Cleanup cancelled")


def main():
    """Enhanced command line interface for model management."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced AI Model Manager')
    parser.add_argument('command', choices=[
        'list', 'organize', 'backup', 'delete', 'info', 'compare', 'cleanup', 'interactive'
    ], help='Command to execute')
    
    parser.add_argument('--model', '-m', help='Model name')
    parser.add_argument('--model2', '-m2', help='Second model name (for compare)')
    parser.add_argument('--notes', '-n', default='', help='Notes to add')
    parser.add_argument('--steps', '-s', type=int, help='Total training steps')
    parser.add_argument('--success-rate', '-r', type=float, help='Success rate percentage')
    parser.add_argument('--days', '-d', type=int, default=30, help='Days to keep for cleanup')
    parser.add_argument('--force', '-f', action='store_true', help='Skip confirmations')
    parser.add_argument('--detailed', action='store_true', help='Show detailed information')
    parser.add_argument('--sort', choices=['name', 'size', 'modified', 'success_rate'], 
                       default='modified', help='Sort models by criteria')
    
    args = parser.parse_args()
    
    manager = ModelManager()
    
    try:
        if args.command == 'list':
            manager.list_models(detailed=args.detailed, sort_by=args.sort)
        
        elif args.command == 'organize':
            manager.organize_models()
        
        elif args.command == 'backup':
            if args.model:
                manager.backup_model(args.model, args.notes)
            else:
                manager.backup_model("", "", interactive=True)
        
        elif args.command == 'delete':
            if args.model:
                manager.safe_delete_model(args.model, args.force)
            else:
                manager.safe_delete_model("", args.force, interactive=True)
        
        elif args.command == 'info':
            if args.model:
                manager.add_model_info(args.model, args.success_rate, args.steps, args.notes)
            else:
                manager.add_model_info("", interactive=True)
        
        elif args.command == 'compare':
            if args.model and args.model2:
                manager.compare_models(args.model, args.model2, args.detailed)
            else:
                print("❌ --model and --model2 required for compare")
        
        elif args.command == 'cleanup':
            manager.interactive_cleanup(args.days)
        
        elif args.command == 'interactive':
            interactive_menu(manager)
            
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def interactive_menu(manager: ModelManager):
    """Interactive menu system for model management."""
    while True:
        print("\n" + "="*50)
        print("🤖 AI Model Manager - Interactive Mode")
        print("="*50)
        print("1. 📋 List models")
        print("2. 🗂️ Organize models")
        print("3. 💾 Backup model")
        print("4. 🗑️ Delete model")
        print("5. 📝 Update model info")
        print("6. ⚖️ Compare models")
        print("7. 🧹 Cleanup old models")
        print("8. ❌ Exit")
        
        try:
            choice = input("\nSelect option (1-8): ").strip()
            
            if choice == '1':
                detailed = input("Show detailed info? (y/N): ").strip().lower() == 'y'
                sort_options = ['modified', 'name', 'size', 'success_rate']
                print(f"Sort options: {', '.join(sort_options)}")
                sort_by = input("Sort by (default: modified): ").strip() or 'modified'
                manager.list_models(detailed=detailed, sort_by=sort_by)
                
            elif choice == '2':
                manager.organize_models()
                
            elif choice == '3':
                manager.backup_model("", "", interactive=True)
                
            elif choice == '4':
                manager.safe_delete_model("", interactive=True)
                
            elif choice == '5':
                manager.add_model_info("", interactive=True)
                
            elif choice == '6':
                model1 = manager.interactive_model_selection("Select first model")
                if model1:
                    model2 = manager.interactive_model_selection("Select second model")
                    if model2:
                        manager.compare_models(model1, model2)
                
            elif choice == '7':
                days_input = input("Keep models newer than how many days? (default: 30): ").strip()
                days = int(days_input) if days_input else 30
                manager.interactive_cleanup(days)
                
            elif choice == '8':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please select 1-8.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except ValueError as e:
            print(f"❌ Invalid input: {e}")
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 