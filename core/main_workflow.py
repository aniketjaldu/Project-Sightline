import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from pipeline.data_fetcher import LabelboxDataFetcher
from pipeline.data_processor import DataProcessor
from core.model_trainer import YOLOModelTrainer
from pipeline.inference_runner import YOLOInferenceRunner
from pipeline.annotation_importer import LabelboxAnnotationImporter
from pipeline.data_manager import DataManager
from core.config import ensure_directories, get_workflow_directories
from utils.logger import setup_logger

class MainWorkflow:
    """Main workflow orchestrator for Labelbox-YOLO pipeline"""
    
    def __init__(self, workflow_id: Optional[str] = None, tracking_method: str = "sightline"):
        self.workflow_id = workflow_id or f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger = setup_logger(self.__class__.__name__)
        self.tracking_method = tracking_method.lower()
        
        # Initialize all components
        self.data_fetcher = LabelboxDataFetcher()
        self.data_processor = DataProcessor(tracking_method=self.tracking_method)
        self.model_trainer = YOLOModelTrainer()
        self.inference_runner = YOLOInferenceRunner()
        self.annotation_importer = LabelboxAnnotationImporter()
        self.data_manager = DataManager()
        
        ensure_directories()
        
        self.logger.info(f"Workflow initialized: {self.workflow_id}")
        self.logger.info(f"Training tracking method: {self.tracking_method.upper()}")
    
    def run_complete_training_workflow(
        self, 
        training_data_row_id: str,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the complete training workflow
        
        Args:
            training_data_row_id: Data row ID for training data
            model_name: Name for the trained model
            
        Returns:
            Dictionary with workflow results
        """
        try:
            self.logger.info(f"Starting complete training workflow for data row: {training_data_row_id}")
            
            if model_name is None:
                model_name = f"model_{self.workflow_id}"
            
            training_results = {
                'workflow_id': self.workflow_id,
                'training_data_row_id': training_data_row_id,
                'model_name': model_name,
                'steps': {},
                'success': False
            }
            
            # Step 1: Fetch training data
            self.logger.info("Step 1: Fetching training data from Labelbox")
            download_result = self.data_fetcher.fetch_and_download_complete(
                training_data_row_id, 
                include_labels=True
            )
            
            if not download_result or 'video' not in download_result or 'json' not in download_result:
                self.logger.error("Failed to download training data")
                training_results['steps']['download'] = {'success': False, 'error': 'Failed to download data'}
                return training_results
            
            training_results['steps']['download'] = {
                'success': True,
                'video_path': str(download_result['video']),
                'json_path': str(download_result['json'])
            }
            
            # Step 2: Process data and convert to YOLO format
            self.logger.info("Step 2: Processing data and converting to YOLO format")
            # Get workflow directories and pass them to data processor
            workflow_dirs = get_workflow_directories(self.workflow_id)
            processing_success = self.data_processor.process_labelbox_data(
                json_path=download_result['json'],
                video_path=download_result['video'],
                dataset_type="train",
                workflow_dirs=workflow_dirs
            )
            
            if not processing_success:
                self.logger.error("Failed to process training data")
                training_results['steps']['processing'] = {'success': False, 'error': 'Failed to process data'}
                return training_results
            
            # Step 3: Creating dataset configuration
            self.logger.info("Step 3: Creating dataset configuration")
            # Use the same workflow_dirs that were used for data processing
            dataset_yaml = self.data_processor.create_dataset_yaml(workflow_dirs)
            
            if not dataset_yaml:
                self.logger.error("Failed to create dataset configuration")
                training_results['error'] = "Dataset configuration failed"
                return training_results
            
            training_results['dataset_yaml'] = str(dataset_yaml)
            
            # Step 4: Split into train/validation sets
            self.logger.info("Step 4: Splitting data into train/validation sets")
            if not self.data_processor.split_train_validation(workflow_dirs):
                self.logger.error("Failed to create train/validation split")
                return training_results
            
            # Step 5: Train the model
            self.logger.info("Step 5: Starting model training")
            model_results = self.model_trainer.train_model(
                dataset_yaml_path=dataset_yaml, 
                model_name=model_name
            )
            
            if not model_results:
                self.logger.error("Failed to train model")
                training_results['steps']['training'] = {'success': False, 'error': 'Training failed'}
                return training_results
            
            training_results['steps']['training'] = {
                'success': True,
                'model_path': str(model_results)
            }
            
            # Step 6: Validate model
            self.logger.info("Step 6: Validating trained model")
            validation_metrics = self.model_trainer.validate_model(
                model_path=model_results,
                dataset_yaml_path=dataset_yaml
            )
            
            training_results['steps']['validation'] = {
                'success': validation_metrics is not None,
                'metrics': validation_metrics
            }
            
            # Step 7: Clean up training data
            self.logger.info("Step 7: Cleaning up training data")
            cleanup_success = self.data_manager.cleanup_training_data(
                data_row_ids=[training_data_row_id],
                keep_models=True
            )
            
            training_results['steps']['cleanup'] = {'success': cleanup_success}
            training_results['success'] = True
            training_results['trained_model_path'] = str(model_results)
            
            self.logger.info(f"Training workflow completed successfully: {model_name}")
            return training_results
            
        except Exception as e:
            self.logger.error(f"Error in training workflow: {str(e)}")
            training_results['steps']['error'] = {'success': False, 'error': str(e)}
            return training_results
    
    def run_complete_inference_workflow(
        self, 
        inference_data_row_id: str,
        model_path: Path,
        import_to_labelbox: bool = True,
        tracking_method: str = "sightline"
    ) -> Dict[str, Any]:
        """
        Run the complete inference workflow
        
        Args:
            inference_data_row_id: Data row ID for inference video
            model_path: Path to the trained model
            import_to_labelbox: Whether to import results back to Labelbox
            
        Returns:
            Dictionary with workflow results
        """
        try:
            self.logger.info(f"Starting inference workflow for data row: {inference_data_row_id}")
            
            results = {
                'workflow_id': self.workflow_id,
                'inference_data_row_id': inference_data_row_id,
                'model_path': str(model_path),
                'steps': {},
                'success': False
            }
            
            # Step 1: Fetch inference video
            self.logger.info("Step 1: Fetching inference video from Labelbox")
            download_result = self.data_fetcher.fetch_and_download_complete(
                inference_data_row_id, 
                include_labels=False
            )
            
            if not download_result or 'video' not in download_result:
                self.logger.error("Failed to download inference video")
                results['steps']['download'] = {'success': False, 'error': 'Failed to download video'}
                return results
            
            results['steps']['download'] = {
                'success': True,
                'video_path': str(download_result['video'])
            }
            
            # Step 2: Run inference with annotated video output
            self.logger.info("Step 2: Running YOLO inference with annotated video output")
            video_name = download_result['video'].stem
            workflow_dirs = get_workflow_directories(self.workflow_id)
            output_video_path = workflow_dirs['inference']['results'] / f"{video_name}_annotated.mp4"
            
            inference_result = self.inference_runner.run_tracking_inference_on_video(
                model_path=model_path,
                video_path=download_result['video'],
                output_video_path=output_video_path,  # Always generate annotated video
                data_row_id=inference_data_row_id,
                tracking_method=tracking_method
            )
            
            if not inference_result:
                self.logger.error("Failed to run inference")
                results['steps']['inference'] = {'success': False, 'error': 'Inference failed'}
                return results
            
            results['steps']['inference'] = {
                'success': True,
                'annotations_file': str(inference_result['annotations_file']),
                'output_video': str(inference_result['output_video']) if inference_result['output_video'] else None,
                'stats': {
                    'total_frames': inference_result.get('total_frames', 0),
                    'tracking_method': tracking_method
                }
            }
            
            # Step 3: Move inference video to dedicated directory
            self.logger.info("Step 3: Moving inference video to dedicated directory")
            moved_videos = self.data_manager.move_inference_videos([download_result['video']])
            results['steps']['video_management'] = {
                'success': len(moved_videos) > 0,
                'moved_video_path': str(moved_videos[0]) if moved_videos else None
            }
            
            # Step 4: Import annotations to Labelbox as labels (if requested)
            if import_to_labelbox:
                self.logger.info("Step 4: Importing annotations to Labelbox as labels")
                import_job_id = self.annotation_importer.import_annotations_from_file(
                    annotations_file=inference_result['annotations_file'],
                    data_row_id=inference_data_row_id
                )
                
                results['steps']['import'] = {
                    'success': import_job_id is not None,
                    'import_job_id': import_job_id
                }
            else:
                results['steps']['import'] = {'success': True, 'skipped': True}
            
            results['success'] = True
            results['annotations_file'] = str(inference_result['annotations_file'])
            if inference_result.get('output_video'):
                results['annotated_video'] = str(inference_result['output_video'])
                self.logger.info(f"Annotated video saved: {inference_result['output_video']}")
            
            self.logger.info("Inference workflow completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in inference workflow: {str(e)}")
            results['steps']['error'] = {'success': False, 'error': str(e)}
            return results
    
    def run_full_pipeline(
        self, 
        training_data_row_id: str,
        inference_data_row_ids: List[str],
        model_name: Optional[str] = None,
        tracking_method: str = "sightline"
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline: training + inference
        
        Args:
            training_data_row_id: Data row ID for training
            inference_data_row_ids: List of data row IDs for inference
            model_name: Name for the trained model
            
        Returns:
            Dictionary with complete pipeline results
        """
        try:
            self.logger.info(f"Starting full pipeline: training + {len(inference_data_row_ids)} inference tasks")
            
            pipeline_results = {
                'workflow_id': self.workflow_id,
                'training_result': None,
                'inference_results': {},
                'success': False
            }
            
            # Run training workflow
            training_result = self.run_complete_training_workflow(
                training_data_row_id=training_data_row_id,
                model_name=model_name
            )
            
            pipeline_results['training_result'] = training_result
            
            if not training_result['success']:
                self.logger.error("Training workflow failed, stopping pipeline")
                return pipeline_results
            
            trained_model_path = Path(training_result['trained_model_path'])
            
            # Run inference workflows
            successful_inferences = 0
            for inference_data_row_id in inference_data_row_ids:
                self.logger.info(f"Running inference for data row: {inference_data_row_id}")
                
                inference_result = self.run_complete_inference_workflow(
                    inference_data_row_id=inference_data_row_id,
                    model_path=trained_model_path,
                    import_to_labelbox=True,
                    tracking_method=tracking_method
                )
                
                pipeline_results['inference_results'][inference_data_row_id] = inference_result
                
                if inference_result['success']:
                    successful_inferences += 1
            
            # Archive workflow
            self.logger.info("Archiving completed workflow")
            all_data_row_ids = [training_data_row_id] + inference_data_row_ids
            archive_success = self.data_manager.archive_completed_workflow(
                workflow_id=self.workflow_id,
                data_row_ids=all_data_row_ids
            )
            
            pipeline_results['archive_success'] = archive_success
            pipeline_results['success'] = successful_inferences == len(inference_data_row_ids)
            pipeline_results['stats'] = {
                'total_inference_tasks': len(inference_data_row_ids),
                'successful_inferences': successful_inferences,
                'success_rate': successful_inferences / len(inference_data_row_ids) if inference_data_row_ids else 0
            }
            
            self.logger.info(f"Full pipeline completed: {successful_inferences}/{len(inference_data_row_ids)} successful")
            return pipeline_results
            
        except Exception as e:
            self.logger.error(f"Error in full pipeline: {str(e)}")
            pipeline_results['error'] = str(e)
            return pipeline_results
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status and statistics"""
        try:
            storage_usage = self.data_manager.get_storage_usage()
            
            status = {
                'workflow_id': self.workflow_id,
                'timestamp': datetime.now().isoformat(),
                'storage_usage': storage_usage,
                'directories': {
                    'downloads': storage_usage.get('downloads', {}).get('formatted', '0 B'),
                    'dataset': storage_usage.get('dataset', {}).get('formatted', '0 B'),
                    'models': storage_usage.get('models', {}).get('formatted', '0 B'),
                    'inference': storage_usage.get('inference', {}).get('formatted', '0 B'),
                    'logs': storage_usage.get('logs', {}).get('formatted', '0 B')
                }
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting workflow status: {str(e)}")
            return {'error': str(e)} 