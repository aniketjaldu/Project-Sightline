#!/usr/bin/env python3

"""
Enhanced script to delete both prelabels (MAL imports) and regular labels for a specific data row in Labelbox.
This version handles both MAL prediction imports and regular labels.
"""

import os
import sys
import labelbox as lb
from pathlib import Path
import time
from datetime import datetime

def load_config():
    """Load configuration from environment variables or config file"""
    # Try to load from environment variables first
    api_key = os.getenv('LABELBOX_API_KEY')
    project_id = os.getenv('LABELBOX_PROJECT_ID')
    
    # If not in environment, try to load from config file
    if not api_key or not project_id:
        config_file = Path('config.py')
        if config_file.exists():
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("config", config_file)
                config = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config)
                
                if not api_key:
                    api_key = getattr(config, 'LABELBOX_API_KEY', None)
                if not project_id:
                    project_id = getattr(config, 'LABELBOX_PROJECT_ID', None)
            except Exception as e:
                print(f"Warning: Could not load config.py: {e}")
    
    return api_key, project_id

def delete_mal_predictions_for_data_row(data_row_id: str, api_key: str = None, project_id: str = None):
    """
    Delete MAL prediction imports (prelabels) for a specific data row.
    
    Args:
        data_row_id: The ID of the data row to delete prelabels for
        api_key: Labelbox API key (optional, will try to load from config)
        project_id: Labelbox project ID (optional, will try to load from config)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load configuration if not provided
        if not api_key or not project_id:
            config_api_key, config_project_id = load_config()
            api_key = api_key or config_api_key
            project_id = project_id or config_project_id
        
        # Validate required parameters
        if not api_key:
            print("❌ Error: LABELBOX_API_KEY not found in environment variables or config.py")
            return False
        
        if not project_id:
            print("❌ Error: LABELBOX_PROJECT_ID not found in environment variables or config.py")
            return False
        
        if not data_row_id:
            print("❌ Error: data_row_id is required")
            return False
        
        print(f"🔄 Initializing Labelbox client...")
        
        # Initialize the Labelbox client
        client = lb.Client(api_key=api_key)
        
        print(f"📂 Getting project: {project_id}")
        project = client.get_project(project_id)
        
        print(f"🔍 Getting data row: {data_row_id}")
        data_row = client.get_data_row(data_row_id)
        
        print(f"🎯 Data row found: {data_row.external_id if hasattr(data_row, 'external_id') else 'N/A'}")
        
        # Get MAL prediction imports for this project
        print(f"📋 Getting MAL prediction imports for project...")
        mal_imports = list(project.get_mal_prediction_imports())
        
        if not mal_imports:
            print(f"ℹ️  No MAL prediction imports found for project {project_id}")
            return True
        
        print(f"🔍 Found {len(mal_imports)} MAL prediction imports")
        
        # Check each MAL import to see if it contains our data row
        deleted_count = 0
        for i, mal_import in enumerate(mal_imports, 1):
            try:
                print(f"🔍 Checking MAL import {i}/{len(mal_imports)}: {mal_import.name}")
                
                # We need to check if this MAL import contains our specific data row
                # Since there's no direct way to filter, we'll check the import details
                # This approach deletes ALL MAL imports that might contain the data row
                
                # Get creation date for reference
                created_at = getattr(mal_import, 'created_at', 'Unknown')
                print(f"  📅 Created: {created_at}")
                
                # For safety, only delete recent imports (you can modify this logic)
                confirmation = input(f"  ⚠️  Delete MAL import '{mal_import.name}'? This will remove ALL prelabels from this import job. (yes/no): ").strip().lower()
                
                if confirmation in ['yes', 'y']:
                    print(f"🗑️  Deleting MAL import: {mal_import.name}")
                    mal_import.delete()
                    deleted_count += 1
                    print(f"✅ Successfully deleted MAL import: {mal_import.name}")
                    time.sleep(0.5)  # Small delay to avoid rate limiting
                else:
                    print(f"⏭️  Skipped MAL import: {mal_import.name}")
                    
            except Exception as e:
                print(f"⚠️  Warning: Could not process MAL import {i}: {str(e)}")
        
        if deleted_count > 0:
            print(f"✅ SUCCESS: Deleted {deleted_count} MAL import(s)")
        else:
            print(f"ℹ️  No MAL imports were deleted")
        
        return deleted_count > 0
        
    except Exception as e:
        print(f"❌ Error deleting MAL imports: {str(e)}")
        return False

def delete_specific_mal_import_by_name(mal_import_name: str, api_key: str = None, project_id: str = None):
    """
    Delete a specific MAL import by name.
    
    Args:
        mal_import_name: The name of the MAL import to delete
        api_key: Labelbox API key (optional, will try to load from config)
        project_id: Labelbox project ID (optional, will try to load from config)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load configuration if not provided
        if not api_key or not project_id:
            config_api_key, config_project_id = load_config()
            api_key = api_key or config_api_key
            project_id = project_id or config_project_id
        
        # Validate required parameters
        if not api_key or not project_id:
            print("❌ Error: Missing API key or project ID")
            return False
        
        print(f"🔄 Initializing Labelbox client...")
        client = lb.Client(api_key=api_key)
        
        print(f"📂 Getting project: {project_id}")
        project = client.get_project(project_id)
        
        print(f"🔍 Looking for MAL import: {mal_import_name}")
        mal_imports = list(project.get_mal_prediction_imports())
        
        # Find the specific import by name
        target_import = None
        for mal_import in mal_imports:
            if mal_import.name == mal_import_name:
                target_import = mal_import
                break
        
        if not target_import:
            print(f"❌ MAL import '{mal_import_name}' not found")
            print(f"📋 Available MAL imports:")
            for mal_import in mal_imports:
                print(f"  - {mal_import.name}")
            return False
        
        print(f"🎯 Found MAL import: {target_import.name}")
        print(f"🗑️  Deleting MAL import...")
        target_import.delete()
        print(f"✅ Successfully deleted MAL import: {mal_import_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error deleting MAL import: {str(e)}")
        return False

def delete_regular_labels_for_data_row(data_row_id: str, api_key: str = None, project_id: str = None):
    """
    Delete regular labels for a specific data row.
    
    Args:
        data_row_id: The ID of the data row to delete labels for
        api_key: Labelbox API key (optional, will try to load from config)
        project_id: Labelbox project ID (optional, will try to load from config)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load configuration if not provided
        if not api_key or not project_id:
            config_api_key, config_project_id = load_config()
            api_key = api_key or config_api_key
            project_id = project_id or config_project_id
        
        # Validate required parameters
        if not api_key:
            print("❌ Error: LABELBOX_API_KEY not found in environment variables or config.py")
            return False
        
        if not project_id:
            print("❌ Error: LABELBOX_PROJECT_ID not found in environment variables or config.py")
            return False
        
        if not data_row_id:
            print("❌ Error: data_row_id is required")
            return False
        
        print(f"🔄 Initializing Labelbox client...")
        
        # Initialize the Labelbox client
        client = lb.Client(api_key=api_key)
        
        print(f"📂 Getting project: {project_id}")
        project = client.get_project(project_id)
        
        print(f"🔍 Getting data row: {data_row_id}")
        data_row = client.get_data_row(data_row_id)
        
        print(f"🎯 Data row found: {data_row.external_id if hasattr(data_row, 'external_id') else 'N/A'}")
        
        # Get labels for this data row
        print(f"📋 Getting labels for data row...")
        labels = list(data_row.labels())
        
        if not labels:
            print(f"ℹ️  No regular labels found for data row {data_row_id}")
            return True
        
        print(f"🔍 Found {len(labels)} regular labels")
        
        # Delete each label
        deleted_count = 0
        for i, label in enumerate(labels, 1):
            try:
                # Get label info
                label_id = getattr(label, 'uid', 'Unknown')
                created_by = getattr(label, 'created_by', None)
                created_by_name = getattr(created_by, 'email', 'Unknown') if created_by else 'Unknown'
                created_at = getattr(label, 'created_at', 'Unknown')
                
                print(f"🔍 Label {i}/{len(labels)}: {label_id}")
                print(f"  📅 Created: {created_at}")
                print(f"  👤 Created by: {created_by_name}")
                
                # For safety, ask for confirmation for each label
                confirmation = input(f"  ⚠️  Delete label '{label_id}'? (yes/no/all): ").strip().lower()
                
                if confirmation in ['yes', 'y']:
                    print(f"🗑️  Deleting label: {label_id}")
                    label.delete()
                    deleted_count += 1
                    print(f"✅ Successfully deleted label: {label_id}")
                    time.sleep(0.5)  # Small delay to avoid rate limiting
                elif confirmation == 'all':
                    # Delete this and all remaining labels
                    print(f"🗑️  Deleting label: {label_id}")
                    label.delete()
                    deleted_count += 1
                    print(f"✅ Successfully deleted label: {label_id}")
                    
                    # Delete remaining labels without asking
                    for j, remaining_label in enumerate(labels[i:], i+1):
                        try:
                            remaining_label_id = getattr(remaining_label, 'uid', 'Unknown')
                            print(f"🗑️  Deleting label: {remaining_label_id}")
                            remaining_label.delete()
                            deleted_count += 1
                            print(f"✅ Successfully deleted label: {remaining_label_id}")
                            time.sleep(0.5)
                        except Exception as e:
                            print(f"⚠️  Warning: Could not delete label {remaining_label_id}: {str(e)}")
                    break
                else:
                    print(f"⏭️  Skipped label: {label_id}")
                    
            except Exception as e:
                print(f"⚠️  Warning: Could not process label {i}: {str(e)}")
        
        if deleted_count > 0:
            print(f"✅ SUCCESS: Deleted {deleted_count} regular label(s)")
        else:
            print(f"ℹ️  No regular labels were deleted")
        
        return deleted_count > 0
        
    except Exception as e:
        print(f"❌ Error deleting regular labels: {str(e)}")
        return False

def delete_all_labels_for_data_row(data_row_id: str, api_key: str = None, project_id: str = None):
    """
    Delete both MAL predictions and regular labels for a specific data row.
    
    Args:
        data_row_id: The ID of the data row to delete all labels for
        api_key: Labelbox API key (optional, will try to load from config)
        project_id: Labelbox project ID (optional, will try to load from config)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"🧹 COMPREHENSIVE CLEANUP for data row: {data_row_id}")
        print("=" * 60)
        
        mal_success = False
        labels_success = False
        
        # Step 1: Delete MAL predictions (prelabels)
        print(f"\n📋 Step 1: Deleting MAL predictions (prelabels)...")
        try:
            mal_success = delete_mal_predictions_for_data_row(data_row_id, api_key, project_id)
        except Exception as e:
            print(f"⚠️  Warning: MAL deletion failed: {str(e)}")
        
        # Step 2: Delete regular labels
        print(f"\n🏷️  Step 2: Deleting regular labels...")
        try:
            labels_success = delete_regular_labels_for_data_row(data_row_id, api_key, project_id)
        except Exception as e:
            print(f"⚠️  Warning: Regular label deletion failed: {str(e)}")
        
        # Summary
        print(f"\n📊 CLEANUP SUMMARY:")
        print(f"  • MAL predictions: {'✅ Deleted' if mal_success else '❌ Failed/None found'}")
        print(f"  • Regular labels: {'✅ Deleted' if labels_success else '❌ Failed/None found'}")
        
        overall_success = mal_success or labels_success
        if overall_success:
            print(f"\n✅ Data row {data_row_id} has been cleaned up successfully!")
        else:
            print(f"\n⚠️  No labels were deleted (they may not have existed)")
        
        return overall_success
        
    except Exception as e:
        print(f"❌ Error in comprehensive cleanup: {str(e)}")
        return False

def main():
    """Main function to handle command line usage"""
    print("🗑️  Enhanced Labelbox Label Deletion Tool")
    print("🔧 Handles both MAL predictions (prelabels) and regular labels")
    print("=" * 60)
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        first_arg = sys.argv[1]
        
        # Check if it looks like a data row ID (starts with 'cm' for Labelbox IDs)
        if first_arg.startswith('cm') and len(first_arg) > 20:
            data_row_id = first_arg
            print(f"📋 Using data_row_id from command line: {data_row_id}")
            
            # Ask what type of deletion to perform
            print(f"\n🤔 What would you like to delete for this data row?")
            print("1. MAL predictions only (prelabels)")
            print("2. Regular labels only")
            print("3. ALL labels (both MAL predictions and regular labels)")
            
            delete_choice = input("Enter choice (1-3): ").strip()
            
            # Confirm deletion
            if delete_choice == "1":
                print(f"\n⚠️  WARNING: This will delete MAL predictions for data row {data_row_id}")
            elif delete_choice == "2":
                print(f"\n⚠️  WARNING: This will delete regular labels for data row {data_row_id}")
            elif delete_choice == "3":
                print(f"\n⚠️  WARNING: This will delete ALL labels (MAL + regular) for data row {data_row_id}")
            else:
                print("❌ Invalid choice")
                return
            
            print("This action cannot be undone.")
            confirm = input("Are you sure you want to continue? (yes/no): ").strip().lower()
            
            if confirm not in ['yes', 'y']:
                print("❌ Operation cancelled by user")
                return
            
            # Perform deletion based on choice
            if delete_choice == "1":
                success = delete_mal_predictions_for_data_row(data_row_id)
            elif delete_choice == "2":
                success = delete_regular_labels_for_data_row(data_row_id)
            elif delete_choice == "3":
                success = delete_all_labels_for_data_row(data_row_id)
        else:
            # Treat as MAL import name
            mal_import_name = first_arg
            print(f"📋 Using MAL import name from command line: {mal_import_name}")
            
            # Confirm deletion
            print(f"\n⚠️  WARNING: This will delete the MAL import: {mal_import_name}")
            print("This action cannot be undone.")
            confirm = input("Are you sure you want to continue? (yes/no): ").strip().lower()
            
            if confirm not in ['yes', 'y']:
                print("❌ Operation cancelled by user")
                return
            
            # Perform deletion
            success = delete_specific_mal_import_by_name(mal_import_name)
    else:
        # Interactive mode
        print("📝 Choose deletion method:")
        print("1. Delete MAL predictions by data row ID")
        print("2. Delete regular labels by data row ID")
        print("3. Delete ALL labels by data row ID (comprehensive cleanup)")
        print("4. Delete specific MAL import by name")
        choice = input("Enter choice (1-4): ").strip()
        
        if choice in ["1", "2", "3"]:
            data_row_id = input("Data Row ID: ").strip()
            if not data_row_id:
                print("❌ Error: No data row ID provided")
                return
            
            if choice == "1":
                print(f"\n⚠️  WARNING: This will delete MAL predictions for data row {data_row_id}")
                confirm = input("Continue? (yes/no): ").strip().lower()
                if confirm in ['yes', 'y']:
                    success = delete_mal_predictions_for_data_row(data_row_id)
                else:
                    print("❌ Operation cancelled")
                    return
            elif choice == "2":
                print(f"\n⚠️  WARNING: This will delete regular labels for data row {data_row_id}")
                confirm = input("Continue? (yes/no): ").strip().lower()
                if confirm in ['yes', 'y']:
                    success = delete_regular_labels_for_data_row(data_row_id)
                else:
                    print("❌ Operation cancelled")
                    return
            elif choice == "3":
                print(f"\n⚠️  WARNING: This will delete ALL labels (MAL + regular) for data row {data_row_id}")
                print("This is a comprehensive cleanup that removes everything.")
                confirm = input("Continue? (yes/no): ").strip().lower()
                if confirm in ['yes', 'y']:
                    success = delete_all_labels_for_data_row(data_row_id)
                else:
                    print("❌ Operation cancelled")
                    return
                    
        elif choice == "4":
            mal_import_name = input("MAL Import Name: ").strip()
            if not mal_import_name:
                print("❌ Error: No MAL import name provided")
                return
            
            print(f"\n⚠️  WARNING: This will delete the MAL import: {mal_import_name}")
            confirm = input("Continue? (yes/no): ").strip().lower()
            if confirm in ['yes', 'y']:
                success = delete_specific_mal_import_by_name(mal_import_name)
            else:
                print("❌ Operation cancelled")
                return
        else:
            print("❌ Error: Invalid choice")
            return
    
    if success:
        print(f"\n🎉 Label deletion completed successfully!")
        print(f"\n💡 TIP: You can now run your workflow again to create fresh annotations")
    else:
        print(f"\n❌ Label deletion failed or no labels were found to delete.")

if __name__ == "__main__":
    main() 