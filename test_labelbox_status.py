#!/usr/bin/env python3
"""
Labelbox Status Management Test Script

This script helps you explore and test moving datarows to different statuses
in your Labelbox project, specifically the "In review" status.

Usage:
    python test_labelbox_status.py <data_row_id>
    python test_labelbox_status.py <data_row_id> --list-queues
    python test_labelbox_status.py <data_row_id> --queue-name "Review Queue"
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import labelbox as lb
from core.config import LABELBOX_API_KEY, LABELBOX_PROJECT_ID
from utils.logger import setup_logger

def list_project_task_queues(project):
    """List all task queues in the project"""
    print("\n📋 Available Task Queues:")
    print("=" * 50)
    
    task_queues = list(project.task_queues())
    
    if not task_queues:
        print("❌ No task queues found in project")
        return []
    
    for i, queue in enumerate(task_queues, 1):
        print(f"{i}. {queue.name}")
        print(f"   ID: {queue.uid}")
        print(f"   Type: {getattr(queue, 'type', 'N/A')}")
        print()
    
    return task_queues

def get_datarow_current_status(client, data_row_id):
    """Get current status/queue of a datarow"""
    try:
        data_row = client.get_data_row(data_row_id)
        print(f"\n🔍 Current DataRow Status:")
        print(f"   ID: {data_row.uid}")
        print(f"   External ID: {getattr(data_row, 'external_id', 'N/A')}")
        print(f"   Global Key: {getattr(data_row, 'global_key', 'N/A')}")
        
        # Check labels
        labels = list(data_row.labels())
        print(f"   Labels: {len(labels)} found")
        
        if labels:
            latest_label = labels[-1]  # Most recent
            print(f"   Latest Label ID: {latest_label.uid}")
            print(f"   Latest Label Created: {getattr(latest_label, 'created_at', 'N/A')}")
        
        return data_row
    except Exception as e:
        print(f"❌ Error getting datarow status: {e}")
        return None

def move_to_review_queue(project, data_row_id, queue_name=None, client=None):
    """Move datarow to review queue"""
    try:
        # Get all task queues
        task_queues = list(project.task_queues())
        
        if not task_queues:
            print("❌ No task queues found")
            return False
        
        # Find review queue
        review_queue = None
        
        if queue_name:
            # Look for exact match first
            for queue in task_queues:
                if queue.name.lower() == queue_name.lower():
                    review_queue = queue
                    break
        
        if not review_queue:
            # Auto-detect review queue
            for queue in task_queues:
                queue_name_lower = queue.name.lower()
                if any(term in queue_name_lower for term in [
                    'review', 'to_review', 'in_review', 'to review', 'in review'
                ]):
                    review_queue = queue
                    break
        
        if not review_queue:
            print("❌ No review queue found")
            print("Available queues:")
            for queue in task_queues:
                print(f"  - {queue.name}")
            return False
        
        print(f"\n🎯 Moving to queue: {review_queue.name}")
        
        # Move datarow to review queue using correct API signature
        # Try different identifier formats to find what works
        try:
            # Method 1: Try with DataRowIds class
            from labelbox.schema.identifiables import DataRowIds
            identifier = DataRowIds([data_row_id])
            result = project.move_data_rows_to_task_queue(
                data_row_ids=identifier,
                task_queue_id=review_queue.uid
            )
        except (ImportError, TypeError) as e1:
            try:
                # Method 2: Try with GlobalKeys
                from labelbox.schema.identifiables import GlobalKeys
                data_row = client.get_data_row(data_row_id)
                if data_row.global_key:
                    identifier = GlobalKeys([data_row.global_key])
                    result = project.move_data_rows_to_task_queue(
                        data_row_ids=identifier,
                        task_queue_id=review_queue.uid
                    )
                else:
                    raise Exception("No global key found")
            except Exception as e2:
                # Method 3: Try direct string list (original approach)
                try:
                    result = project.move_data_rows_to_task_queue(
                        data_row_ids=[data_row_id],
                        task_queue_id=review_queue.uid
                    )
                except Exception as e3:
                    raise Exception(f"All methods failed: {e1}, {e2}, {e3}")
        
        print(f"✅ Successfully moved datarow {data_row_id} to '{review_queue.name}' queue")
        return True
        
    except Exception as e:
        print(f"❌ Error moving to review queue: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test Labelbox status management")
    parser.add_argument("data_row_id", help="Data row ID to test")
    parser.add_argument("--list-queues", action="store_true", 
                       help="List all task queues in project")
    parser.add_argument("--queue-name", type=str,
                       help="Specific queue name to move to")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without making changes")
    
    args = parser.parse_args()
    
    # Initialize Labelbox client
    if not LABELBOX_API_KEY:
        print("❌ LABELBOX_API_KEY not set in environment")
        return 1
    
    if not LABELBOX_PROJECT_ID:
        print("❌ LABELBOX_PROJECT_ID not set in environment")
        return 1
    
    client = lb.Client(api_key=LABELBOX_API_KEY)
    project = client.get_project(LABELBOX_PROJECT_ID)
    
    print(f"🚀 Connected to project: {project.name}")
    print(f"📄 Testing datarow: {args.data_row_id}")
    
    # Get current status
    datarow = get_datarow_current_status(client, args.data_row_id)
    if not datarow:
        return 1
    
    # List queues if requested
    if args.list_queues:
        task_queues = list_project_task_queues(project)
    
    # Move to review queue
    if not args.dry_run:
        print(f"\n🔄 Attempting to move to review status...")
        success = move_to_review_queue(project, args.data_row_id, args.queue_name, client)
        
        if success:
            print(f"\n✅ SUCCESS: Datarow moved to review status")
        else:
            print(f"\n❌ FAILED: Could not move datarow to review status")
            return 1
    else:
        print(f"\n🔍 DRY RUN: Would attempt to move {args.data_row_id} to review status")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 