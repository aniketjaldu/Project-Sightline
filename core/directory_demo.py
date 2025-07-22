#!/usr/bin/env python3
"""
Directory Structure Demonstration & Migration Tool

This script demonstrates the new organized directory structure and provides
migration tools to move from the old flat structure to the new organized one.

Author: AI Assistant
"""

import json
import shutil
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from core.config import (
    get_workflow_directories, ensure_directories, 
    WORKFLOWS_DIR, MODELS_ACTIVE, TEMP_DIR, DATA_ROOT
)
from utils.workflow_manager import WorkflowManager
from utils.logger import setup_logger

def demonstrate_new_structure():
    """Demonstrate the new organized directory structure"""
    print("\n🎯 NEW ORGANIZED DIRECTORY STRUCTURE")
    print("=" * 60)
    
    # Create base directories
    ensure_directories()
    
    # Show the organized structure
    structure = f"""
project_data/                              # Root data directory
├── workflows/                             # Individual workflow runs
│   ├── training_20250122_143022/         # Timestamped workflow
│   │   ├── workflow.json                 # Workflow manifest
│   │   ├── inputs/                       # Raw input data
│   │   │   ├── videos/                   # Original videos
│   │   │   └── annotations/              # Original JSON annotations
│   │   ├── training/                     # Training-specific data
│   │   │   ├── dataset/                  # YOLO dataset
│   │   │   │   ├── train/
│   │   │   │   │   ├── images/          # Training images
│   │   │   │   │   └── labels/          # Training labels
│   │   │   │   └── val/
│   │   │   │       ├── images/          # Validation images
│   │   │   │       └── labels/          # Validation labels
│   │   │   ├── configs/                 # Training configs
│   │   │   ├── checkpoints/             # Training checkpoints
│   │   │   └── results/                 # Training results
│   │   ├── temp/                        # Temporary processing files
│   │   ├── outputs/                     # Final outputs
│   │   │   ├── models/                  # Trained model files
│   │   │   ├── videos/                  # Output videos
│   │   │   ├── annotations/             # Output annotations
│   │   │   └── reports/                 # Analysis reports
│   │   └── logs/                        # Workflow-specific logs
│   └── inference_20250122_143055/       # Inference workflow
│       ├── inputs/                      # Input videos
│       ├── inference/                   # Inference data
│       ├── tracking/                    # Tracking method outputs
│       │   ├── sightline/              # Sightline tracking results
│       │   ├── bytetrack/              # ByteTrack tracking results
│       │   ├── botsort/                # BoT-SORT tracking results
│       │   └── comparisons/            # Method comparisons
│       └── outputs/                    # Final outputs
├── models/                             # Permanent model storage
│   ├── active/                         # Currently active models
│   │   ├── my_model.pt                # Active model file
│   │   └── my_model_info.json         # Model metadata
│   ├── versions/                       # Model version history
│   └── pretrained/                     # Downloaded pretrained models
├── configs/                            # Configuration files
│   ├── trackers/                       # Tracking configurations
│   ├── datasets/                       # Dataset configurations
│   └── models/                         # Model configurations
├── temp/                              # Global temporary files
│   ├── downloads/                      # Raw downloads (can be deleted)
│   ├── processing/                     # Processing workspace
│   └── inference/                      # Inference workspace
├── logs/                              # Centralized logging
│   ├── main.log                       # Main application log
│   ├── data_rows.log                  # Data row actions log
│   └── errors.log                     # Error-specific log
└── archive/                           # Completed workflows
    └── training_20250120_120000/      # Archived workflow
"""
    
    print(structure)
    print("\n📋 KEY BENEFITS:")
    print("✅ Clear separation of concerns")
    print("✅ Easy to find files from specific workflows") 
    print("✅ Temporary files clearly separated from permanent")
    print("✅ Tracking method outputs organized")
    print("✅ Timestamped workflows for easy identification")
    print("✅ Models properly versioned and managed")
    print("✅ Easy cleanup of old workflows")

def demonstrate_workflow_creation():
    """Demonstrate creating a new workflow"""
    print("\n🔧 WORKFLOW CREATION DEMONSTRATION")
    print("=" * 60)
    
    manager = WorkflowManager()
    
    # Create a training workflow
    print("Creating training workflow...")
    training_workflow = manager.create_workflow('training', 'demo_training')
    
    if training_workflow['success']:
        print(f"✅ Created training workflow: {training_workflow['workflow_id']}")
        print(f"📁 Workflow directory: {training_workflow['directories']['root']}")
        
        # Show some key directories
        dirs = training_workflow['directories']
        print(f"📝 Training dataset: {dirs['training']['dataset']}")
        print(f"🏁 Output models: {dirs['outputs']['models']}")
        print(f"📊 Logs: {dirs['logs']['root']}")
    else:
        print(f"❌ Failed to create training workflow: {training_workflow.get('error', 'Unknown error')}")
    
    # Create an inference workflow
    print("\nCreating inference workflow...")
    inference_workflow = manager.create_workflow('inference', 'demo_inference')
    
    if inference_workflow['success']:
        print(f"✅ Created inference workflow: {inference_workflow['workflow_id']}")
        dirs = inference_workflow['directories']
        print(f"🎬 Input videos: {dirs['inputs']['videos']}")
        print(f"🎯 Sightline tracking: {dirs['tracking']['sightline']}")
        print(f"🎯 ByteTrack tracking: {dirs['tracking']['bytetrack']}")
        print(f"🎯 BoT-SORT tracking: {dirs['tracking']['botsort']}")

def demonstrate_file_organization():
    """Demonstrate organized file storage"""
    print("\n📁 FILE ORGANIZATION DEMONSTRATION")  
    print("=" * 60)
    
    manager = WorkflowManager()
    
    # Create demo workflow
    workflow = manager.create_workflow('inference', 'file_demo')
    
    if workflow['success']:
        dirs = workflow['directories']
        
        # Simulate saving different types of files
        demo_files = {
            'input_video': ('sample_video.mp4', 'Demo video content'),
            'input_annotation': ('sample_annotations.json', {'demo': 'data'}),
            'tracking_sightline': ('video_annotations.json', {'tracking': 'sightline_data'}),
            'tracking_bytetrack': ('video_annotations.json', {'tracking': 'bytetrack_data'}),
            'output_video': ('final_video.mp4', 'Final processed video'),
            'output_annotation': ('final_annotations.json', {'final': 'results'})
        }
        
        saved_paths = {}
        for file_type, (filename, content) in demo_files.items():
            path = manager.save_workflow_file(dirs, file_type, content, filename)
            if path:
                saved_paths[file_type] = path
                print(f"✅ Saved {file_type}: {path}")
        
        print(f"\n📋 Summary: Saved {len(saved_paths)} files in organized structure")

def migrate_old_structure():
    """Migration tool from old flat structure to new organized structure"""
    print("\n🔄 MIGRATION FROM OLD STRUCTURE")
    print("=" * 60)
    
    # Check if old structure exists
    old_workflow_data = Path("workflow_data")
    
    if not old_workflow_data.exists():
        print("ℹ️ No old 'workflow_data' directory found - nothing to migrate")
        return
    
    print(f"📁 Found old structure at: {old_workflow_data}")
    
    # Ask for confirmation
    response = input("\n⚠️ Do you want to migrate to the new structure? (y/N): ").strip().lower()
    if response != 'y':
        print("Migration cancelled")
        return
    
    logger = setup_logger("Migration")
    manager = WorkflowManager()
    
    try:
        # Create migration workflow
        migration = manager.create_workflow('migration', 'structure_migration')
        
        if not migration['success']:
            print(f"❌ Failed to create migration workflow: {migration.get('error')}")
            return
        
        migration_dir = migration['directories']['root']
        migrated_files = 0
        
        # Migrate downloads
        old_downloads = old_workflow_data / "downloads"
        if old_downloads.exists():
            new_downloads = TEMP_DIR / "downloads"
            print(f"🔄 Migrating downloads: {old_downloads} -> {new_downloads}")
            
            for file_path in old_downloads.iterdir():
                if file_path.is_file():
                    dest_path = new_downloads / file_path.name
                    shutil.copy2(file_path, dest_path)
                    migrated_files += 1
        
        # Migrate models to active models
        old_models = old_workflow_data / "models"
        if old_models.exists():
            print(f"🔄 Migrating models: {old_models} -> {MODELS_ACTIVE}")
            
            for file_path in old_models.rglob("*.pt"):
                if file_path.is_file():
                    # Try to determine model name from path
                    if file_path.parent.name == "weights" and file_path.name == "best.pt":
                        model_name = file_path.parent.parent.name
                    else:
                        model_name = file_path.stem
                    
                    dest_path = MODELS_ACTIVE / f"{model_name}.pt"
                    if not dest_path.exists():  # Don't overwrite existing
                        shutil.copy2(file_path, dest_path)
                        migrated_files += 1
                        print(f"✅ Migrated model: {model_name}")
        
        # Create migration report
        migration_report = {
            'migration_date': datetime.now().isoformat(),
            'old_structure': str(old_workflow_data),
            'new_structure': str(DATA_ROOT),
            'files_migrated': migrated_files,
            'migration_workflow': migration['workflow_id']
        }
        
        report_path = migration_dir / 'outputs' / 'migration_report.json'
        with open(report_path, 'w') as f:
            json.dump(migration_report, f, indent=2)
        
        print(f"\n✅ Migration completed!")
        print(f"📊 Files migrated: {migrated_files}")
        print(f"📋 Migration report: {report_path}")
        print(f"\n💡 You can now safely archive or remove the old 'workflow_data' directory")
        
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        print(f"❌ Migration failed: {str(e)}")

def show_current_status():
    """Show current directory status"""
    print("\n📊 CURRENT DIRECTORY STATUS")
    print("=" * 60)
    
    manager = WorkflowManager()
    summary = manager.get_workflow_summary()
    
    if 'error' in summary:
        print(f"❌ Error getting status: {summary['error']}")
        return
    
    print(f"🏃 Active workflows: {summary['active_workflows']}")
    print(f"📦 Archived workflows: {summary['archived_workflows']}")
    print(f"🤖 Total models: {summary['total_models']}")
    print(f"🗂️ Temp directory size: {summary['temp_size_mb']} MB")
    
    if summary['workflows']:
        print("\n📋 Recent workflows:")
        for workflow in summary['workflows']:
            print(f"  • {workflow['id']} ({workflow['type']}) - {workflow.get('created', 'Unknown date')}")

def main():
    """Main demonstration function"""
    parser = argparse.ArgumentParser(description="Directory Structure Demo & Migration Tool")
    parser.add_argument('--action', choices=['demo', 'migrate', 'status', 'all'], 
                       default='all', help='Action to perform')
    
    args = parser.parse_args()
    
    print("🚀 PROJECT SIGHTLINE - DIRECTORY STRUCTURE TOOL")
    print("=" * 60)
    
    if args.action in ['demo', 'all']:
        demonstrate_new_structure()
        demonstrate_workflow_creation()
        demonstrate_file_organization()
    
    if args.action in ['migrate', 'all']:
        migrate_old_structure()
    
    if args.action in ['status', 'all']:
        show_current_status()
    
    print("\n✨ Directory structure demonstration complete!")
    print("\n💡 Next steps:")
    print("  • Update your workflow scripts to use the new WorkflowManager")
    print("  • Run migration if you have old data to preserve")  
    print("  • Enjoy the organized, intuitive file structure! 🎉")

if __name__ == "__main__":
    main() 