#!/usr/bin/env python3
"""
Labelbox-YOLO Workflow Runner

This script provides a command-line interface for running the complete
Labelbox-YOLO workflow including training, inference, and annotation import.

Usage Examples:
    # Run complete pipeline (training + inference)
    python run_workflow.py --mode full --training-id TRAIN_ID --inference-ids INF_ID1 INF_ID2 --model-name my_model

    # Run only training
    python run_workflow.py --mode training --training-id TRAIN_ID --model-name my_model

    # Run only inference (requires existing model)
    python run_workflow.py --mode inference --inference-ids INF_ID1 INF_ID2 --model-path path/to/model.pt

    # Get workflow status
    python run_workflow.py --mode status
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List

from core.main_workflow import MainWorkflow
from core.config import ensure_directories, MODELS_DIR
from utils.logger import setup_logger

def validate_data_row_ids(data_row_ids: List[str]) -> bool:
    """Validate that data row IDs are in correct format"""
    for data_row_id in data_row_ids:
        if not isinstance(data_row_id, str) or len(data_row_id) < 10:
            return False
    return True

def find_model_file(model_name_or_path: str) -> Path:
    """Find model file by name or path with comprehensive search"""
    model_path = Path(model_name_or_path)
    
    # If it's an absolute path and exists, use it
    if model_path.is_absolute() and model_path.exists():
        return model_path
    
    # Try as relative path
    if model_path.exists():
        return model_path
    
    # Comprehensive search in multiple locations (same as training logic)
    possible_paths = [
        MODELS_DIR / "active" / f"{model_name_or_path}.pt",  # Standard location
        MODELS_DIR / model_name_or_path / "weights" / "best.pt",  # Training output location
        MODELS_DIR / model_name_or_path / "weights" / "last.pt",  # Fallback training location
        Path(f"{model_name_or_path}.pt"),  # Current directory with .pt
        Path(model_name_or_path) if model_name_or_path.endswith('.pt') else Path(f"{model_name_or_path}.pt")  # Handle both cases
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    raise FileNotFoundError(f"Model not found: {model_name_or_path}. Searched in: {[str(p) for p in possible_paths]}")

def run_training_workflow(args, workflow: MainWorkflow) -> bool:
    """Run training workflow"""
    logger = setup_logger("TRAINING_WORKFLOW")
    
    logger.info(f"Starting training workflow with data row: {args.training_id}")
    
    result = workflow.run_complete_training_workflow(
        training_data_row_id=args.training_id,
        model_name=args.model_name
    )
    
    if result['success']:
        logger.info("Training workflow completed successfully!")
        logger.info(f"Trained model saved at: {result['trained_model_path']}")
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING WORKFLOW COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Model Name: {result['model_name']}")
        print(f"Model Path: {result['trained_model_path']}")
        print(f"Training Data Row ID: {result['training_data_row_id']}")
        
        if 'validation' in result['steps'] and result['steps']['validation']['metrics']:
            print(f"Validation Metrics: {result['steps']['validation']['metrics']}")
        
        print("="*60)
        return True
    else:
        logger.error("Training workflow failed!")
        print("\n" + "="*60)
        print("TRAINING WORKFLOW FAILED")
        print("="*60)
        for step, step_result in result['steps'].items():
            if not step_result.get('success', True):
                print(f"Failed at step: {step}")
                if 'error' in step_result:
                    print(f"Error: {step_result['error']}")
        print("="*60)
        return False

def run_comparison_workflow(args, workflow: MainWorkflow) -> bool:
    """Run tracking method comparison workflow"""
    logger = setup_logger("COMPARISON_WORKFLOW")
    
    # Validate arguments
    if not args.output_dir:
        logger.error("--output-dir is required for compare mode")
        return False
    
    if not args.inference_ids:
        logger.error("--inference-ids is required for compare mode")
        return False
    
    # Find model
    try:
        if args.model_path:
            model_path = find_model_file(args.model_path)
        elif args.model_name:
            model_path = find_model_file(args.model_name)
        else:
            logger.error("Either --model-path or --model-name must be specified for comparison")
            return False
    except FileNotFoundError as e:
        logger.error(str(e))
        return False
    
    from enhanced_inference_runner import EnhancedInferenceRunner
    
    # Get comparison methods
    comparison_methods = args.comparison_methods
    output_dir = Path(args.output_dir)
    
    logger.info(f"Using model: {model_path}")
    logger.info(f"Comparison methods: {comparison_methods}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Comparing {len(args.inference_ids)} videos")
    
    runner = EnhancedInferenceRunner()
    successful_comparisons = 0
    all_results = {}
    
    for data_row_id in args.inference_ids:
        logger.info(f"Processing comparison for data row: {data_row_id}")
        
        # Download video for comparison
        try:
            download_result = workflow.data_fetcher.fetch_and_download_complete(data_row_id)
            if not download_result or 'video' not in download_result:
                logger.error(f"Failed to download video for {data_row_id}")
                continue
            
            video_path = download_result['video']
            video_name = Path(video_path).stem
            comparison_output_dir = output_dir / f"{video_name}_comparison"
            
            # Run comparison
            comparison_result = runner.compare_tracking_methods(
                model_path=model_path,
                video_path=video_path,
                output_dir=comparison_output_dir,
                methods=comparison_methods,
                data_row_id=data_row_id
            )
            
            all_results[data_row_id] = comparison_result
            
            # Check if comparison was successful
            successful_methods = sum(1 for result in comparison_result.values() if result.get('success', False))
            if successful_methods > 0:
                successful_comparisons += 1
                logger.info(f"Comparison completed for {data_row_id}: {successful_methods}/{len(comparison_methods)} methods successful")
            else:
                logger.error(f"All comparison methods failed for {data_row_id}")
                
        except Exception as e:
            logger.error(f"Error during comparison for {data_row_id}: {str(e)}")
            continue
    
    # Print summary
    print("\n" + "="*60)
    print("TRACKING COMPARISON COMPLETED")
    print("="*60)
    print(f"Model Used: {model_path}")
    print(f"Methods Compared: {', '.join(comparison_methods)}")
    print(f"Output Directory: {output_dir}")
    print(f"Successful Comparisons: {successful_comparisons}/{len(args.inference_ids)}")
    print(f"Success Rate: {(successful_comparisons/len(args.inference_ids))*100:.1f}%")
    
    # Detailed results per video
    for data_row_id, results in all_results.items():
        print(f"\n📹 {data_row_id}:")
        for method, result in results.items():
            if result.get('success'):
                annotations = result.get('annotations', {})
                total_detections = sum(len(frame_annotations) for frame_annotations in annotations.get('frames', {}).values())
                print(f"  ✅ {method.upper()}: {total_detections} detections, {result.get('total_frames', 0)} frames")
                if result.get('output_video'):
                    print(f"      📹 Video: {result['output_video']}")
            else:
                print(f"  ❌ {method.upper()}: Failed - {result.get('error', 'Unknown error')}")
    
    print("="*60)
    return successful_comparisons == len(args.inference_ids)

def run_inference_workflow(args, workflow: MainWorkflow) -> bool:
    """Run inference workflow"""
    logger = setup_logger("INFERENCE_WORKFLOW")
    
    # Find model
    try:
        if args.model_path:
            model_path = find_model_file(args.model_path)
        elif args.model_name:
            model_path = find_model_file(args.model_name)
        else:
            logger.error("Either --model-path or --model-name must be specified for inference")
            return False
    except FileNotFoundError as e:
        logger.error(str(e))
        return False
    
    # Get tracking method
    tracking_method = getattr(args, 'tracking_method', 'sightline')
    
    logger.info(f"Using model: {model_path}")
    logger.info(f"Tracking method: {tracking_method.upper()}")
    logger.info(f"Running inference on {len(args.inference_ids)} videos")
    
    successful_inferences = 0
    results = {}
    
    for data_row_id in args.inference_ids:
        logger.info(f"Processing inference for data row: {data_row_id}")
        
        result = workflow.run_complete_inference_workflow(
            inference_data_row_id=data_row_id,
            model_path=model_path,
            import_to_labelbox=not args.no_import,
            tracking_method=tracking_method
        )
        
        results[data_row_id] = result
        if result['success']:
            successful_inferences += 1
            logger.info(f"Inference completed successfully for {data_row_id}")
        else:
            logger.error(f"Inference failed for {data_row_id}")
    
    # Print summary
    print("\n" + "="*60)
    print("INFERENCE WORKFLOW COMPLETED")
    print("="*60)
    print(f"Model Used: {model_path}")
    print(f"Tracking Method: {tracking_method.upper()}")
    print(f"Successful Inferences: {successful_inferences}/{len(args.inference_ids)}")
    print(f"Success Rate: {(successful_inferences/len(args.inference_ids))*100:.1f}%")
    
    for data_row_id, result in results.items():
        status = "✓" if result['success'] else "✗"
        print(f"{status} {data_row_id}")
        if result['success']:
            print(f"    Annotations saved: {result.get('annotations_file', 'N/A')}")
            if 'steps' in result and 'import' in result['steps']:
                import_result = result['steps']['import']
                if import_result.get('success') and not import_result.get('skipped'):
                    print(f"    Labelbox Label ID: {import_result.get('label_id', 'N/A')}")
    
    print("="*60)
    return successful_inferences == len(args.inference_ids)

def run_full_pipeline(args, workflow: MainWorkflow) -> bool:
    """Run full pipeline (training + inference)"""
    logger = setup_logger("FULL_PIPELINE")
    
    logger.info("Starting full pipeline: training + inference")
    
    result = workflow.run_full_pipeline(
        training_data_row_id=args.training_id,
        inference_data_row_ids=args.inference_ids,
        model_name=args.model_name,
        tracking_method=getattr(args, 'tracking_method', 'sightline')
    )
    
    if result['success']:
        logger.info("Full pipeline completed successfully!")
        
        # Print comprehensive summary
        print("\n" + "="*60)
        print("FULL PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        
        # Training summary
        training_result = result['training_result']
        print(f"Training Data Row ID: {training_result['training_data_row_id']}")
        print(f"Model Name: {training_result['model_name']}")
        print(f"Model Path: {training_result['trained_model_path']}")
        
        # Inference summary
        stats = result['stats']
        print(f"\nInference Results:")
        print(f"Total Tasks: {stats['total_inference_tasks']}")
        print(f"Successful: {stats['successful_inferences']}")
        print(f"Success Rate: {stats['success_rate']*100:.1f}%")
        
        print("\nInference Details:")
        for data_row_id, inf_result in result['inference_results'].items():
            status = "✓" if inf_result['success'] else "✗"
            print(f"{status} {data_row_id}")
        
        print("="*60)
        return True
    else:
        logger.error("Full pipeline failed!")
        print("\n" + "="*60)
        print("FULL PIPELINE FAILED")
        print("="*60)
        
        # Show where it failed
        if result.get('training_result') and not result['training_result']['success']:
            print("Failed at: Training stage")
        else:
            failed_inferences = [
                data_row_id for data_row_id, inf_result in result.get('inference_results', {}).items()
                if not inf_result['success']
            ]
            if failed_inferences:
                print(f"Failed at: Inference stage ({len(failed_inferences)} failures)")
                for data_row_id in failed_inferences:
                    print(f"  - {data_row_id}")
        
        print("="*60)
        return False

def show_workflow_status(workflow: MainWorkflow):
    """Show current workflow status"""
    status = workflow.get_workflow_status()
    
    print("\n" + "="*60)
    print("WORKFLOW STATUS")
    print("="*60)
    print(f"Workflow ID: {status.get('workflow_id', 'N/A')}")
    print(f"Timestamp: {status.get('timestamp', 'N/A')}")
    
    if 'directories' in status:
        print("\nStorage Usage:")
        for dir_name, size in status['directories'].items():
            print(f"  {dir_name.capitalize()}: {size}")
        
        if 'storage_usage' in status and 'total' in status['storage_usage']:
            print(f"  Total: {status['storage_usage']['total']['formatted']}")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description="Labelbox-YOLO Workflow Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['training', 'inference', 'full', 'status', 'compare'],
        help='Workflow mode to run'
    )
    
    parser.add_argument(
        '--training-id',
        type=str,
        help='Data row ID for training data (required for training and full modes)'
    )
    
    parser.add_argument(
        '--inference-ids',
        type=str,
        nargs='+',
        help='Data row IDs for inference videos (required for inference and full modes)'
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        help='Name for the trained model (for training) or existing model name (for inference)'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to existing trained model (for inference mode)'
    )
    
    parser.add_argument(
        '--no-import',
        action='store_true',
        help='Skip importing annotations back to Labelbox'
    )
    
    parser.add_argument(
        '--tracking-method',
        type=str,
        choices=['sightline', 'bytetrack', 'botsort'],
        default='sightline',
        help='Tracking method to use (default: sightline)'
    )
    
    parser.add_argument(
        '--comparison-methods',
        type=str,
        nargs='+',
        choices=['sightline', 'bytetrack', 'botsort'],
        default=['sightline', 'bytetrack', 'botsort'],
        help='Tracking methods to compare (default: all methods)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for comparison results (required for compare mode)'
    )
    
    parser.add_argument(
        '--workflow-id',
        type=str,
        help='Custom workflow ID (optional)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode in ['full', 'training']:
        if not args.training_id:
            print("Error: --training-id is required for training and full modes")
            sys.exit(1)
    
    if args.mode in ['full', 'inference']:
        if not args.inference_ids:
            print("Error: --inference-ids is required for inference and full modes")
            sys.exit(1)
        
        if not validate_data_row_ids(args.inference_ids):
            print("Error: Invalid data row ID format")
            sys.exit(1)
    
    if args.mode == 'inference':
        if not args.model_path and not args.model_name:
            print("Error: Either --model-path or --model-name is required for inference mode")
            sys.exit(1)
    
    # Initialize workflow
    ensure_directories()
    tracking_method = getattr(args, 'tracking_method', 'sightline')
    workflow = MainWorkflow(workflow_id=args.workflow_id, tracking_method=tracking_method)
    
    # Run workflow based on mode
    try:
        if args.mode == 'training':
            success = run_training_workflow(args, workflow)
        elif args.mode == 'inference':
            success = run_inference_workflow(args, workflow)
        elif args.mode == 'full':
            success = run_full_pipeline(args, workflow)
        elif args.mode == 'status':
            show_workflow_status(workflow)
            success = True
        elif args.mode == 'compare':
            success = run_comparison_workflow(args, workflow)
        
        if success:
            print("\n🎉 Workflow completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Workflow failed. Check logs for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Workflow interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 