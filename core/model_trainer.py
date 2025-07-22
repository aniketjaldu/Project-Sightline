import json
from pathlib import Path
from typing import Optional
from ultralytics import YOLO
from core.config import YOLO_CONFIG, MODELS_DIR, DATASET_DIR, ensure_directories
from utils.logger import setup_logger

class YOLOModelTrainer:
    """Class to handle YOLO model training"""
    
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        ensure_directories()
    
    def _find_starting_model(self, model_name: str) -> str:
        """
        Find the best starting model for training based on model name
        
        Args:
            model_name: Name of the model to train
            
        Returns:
            Path to the starting model (either existing trained model or base pretrained model)
        """
        # Normalize model name (remove .pt extension if present for consistent searching)
        base_name = model_name.replace('.pt', '') if model_name.endswith('.pt') else model_name
        
        # Check for existing trained model in multiple locations
        possible_paths = [
            MODELS_DIR / f"{base_name}.pt",  # Standard location
            MODELS_DIR / base_name / "weights" / "best.pt",  # Training output location
            MODELS_DIR / base_name / "weights" / "last.pt",  # Fallback training location
            Path(f"{base_name}.pt"),  # Current directory
            Path(f"{model_name}.pt") if not model_name.endswith('.pt') else Path(model_name),  # Handle exact input
            Path(model_name)  # Try exact input as-is
        ]
        
        for model_path in possible_paths:
            if model_path.exists():
                self.logger.info(f"Found existing model for '{model_name}' at: {model_path}")
                self.logger.info("Using existing model for transfer learning (continuing training)")
                return str(model_path)
        
        # No existing model found, use base pretrained model
        base_model = YOLO_CONFIG['pretrained_model']
        self.logger.info(f"No existing model found for '{model_name}', using base pretrained model: {base_model}")
        return base_model
        
    def train_model(self, dataset_yaml_path: Path, model_name: str = "trained_model") -> Optional[Path]:
        """
        Train YOLO model on the prepared dataset with smart model selection
        
        If a model with the given name already exists, it will be used as the starting point
        for transfer learning. Otherwise, the base pretrained model will be used.
        
        Args:
            dataset_yaml_path: Path to the dataset YAML configuration
            model_name: Name for the trained model (also used to search for existing models)
            
        Returns:
            Path to the trained model file
        """
        try:
            self.logger.info("Starting YOLO model training")
            
            # Verify dataset YAML exists
            if not dataset_yaml_path.exists():
                self.logger.error(f"Dataset YAML not found: {dataset_yaml_path}")
                return None
            
            # Determine which model to use as starting point
            starting_model = self._find_starting_model(model_name)
            self.logger.info(f"Loading starting model: {starting_model}")
            model = YOLO(starting_model)
            
            # Prepare training arguments
            train_args = {
                'data': str(dataset_yaml_path),
                'epochs': YOLO_CONFIG['epochs'],
                'imgsz': YOLO_CONFIG['imgsz'],
                'batch': YOLO_CONFIG['batch'],
                'device': YOLO_CONFIG['device'],
                'project': str(MODELS_DIR),
                'name': model_name,
                'save': True,
                'save_period': 10,  # Save every 10 epochs
                'patience': 20,     # Early stopping patience
                'verbose': True,
                'max_det': YOLO_CONFIG['max_det']  # Valorant optimization: limit detections per image
            }
            
            self.logger.info(f"Training configuration: {train_args}")
            
            # Start training
            results = model.train(**train_args)
            
            # Get the path to the best model
            model_dir = MODELS_DIR / model_name
            best_model_path = model_dir / 'weights' / 'best.pt'
            last_model_path = model_dir / 'weights' / 'last.pt'
            
            # Save model to a standard location
            final_model_path = MODELS_DIR / f"{model_name}.pt"
            
            if best_model_path.exists():
                # Load and save the best model
                best_model = YOLO(str(best_model_path))
                best_model.save(str(final_model_path))
                self.logger.info(f"Best model saved to: {final_model_path}")
            elif last_model_path.exists():
                # Fallback to last model
                last_model = YOLO(str(last_model_path))
                last_model.save(str(final_model_path))
                self.logger.info(f"Last model saved to: {final_model_path}")
            else:
                self.logger.error("No trained model weights found")
                return None
            
            # Save training results
            self._save_training_results(results, model_dir, model_name)
            
            self.logger.info("Model training completed successfully")
            return final_model_path
            
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            return None
    
    def _save_training_results(self, results, model_dir: Path, model_name: str):
        """Save training results and metrics"""
        try:
            results_path = MODELS_DIR / f"{model_name}_results.json"
            
            # Extract key metrics
            training_results = {
                'model_name': model_name,
                'training_config': YOLO_CONFIG,
                'results_dir': str(model_dir),
                'metrics': {}
            }
            
            # Try to extract metrics from results
            if hasattr(results, 'results_dict'):
                training_results['metrics'] = results.results_dict
            
            with open(results_path, 'w') as f:
                json.dump(training_results, f, indent=2, default=str)
            
            self.logger.info(f"Training results saved to: {results_path}")
            
        except Exception as e:
            self.logger.warning(f"Could not save training results: {str(e)}")
    
    def validate_model(self, model_path: Path, dataset_yaml_path: Path) -> Optional[dict]:
        """
        Validate the trained model on the validation set
        
        Args:
            model_path: Path to the trained model
            dataset_yaml_path: Path to the dataset YAML configuration
            
        Returns:
            Dictionary with validation metrics
        """
        try:
            self.logger.info("Validating trained model")
            
            # Load the trained model
            model = YOLO(str(model_path))
            
            # Run validation
            val_results = model.val(
                data=str(dataset_yaml_path),
                device=YOLO_CONFIG['device'],
                verbose=True
            )
            
            # Extract validation metrics
            metrics = {}
            if hasattr(val_results, 'results_dict'):
                metrics = val_results.results_dict
            
            self.logger.info(f"Validation completed. Metrics: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error during model validation: {str(e)}")
            return None
    
    def export_model(self, model_path: Path, export_format: str = 'onnx') -> Optional[Path]:
        """
        Export model to different formats
        
        Args:
            model_path: Path to the trained model
            export_format: Export format ('onnx', 'torchscript', 'coreml', etc.)
            
        Returns:
            Path to the exported model
        """
        try:
            self.logger.info(f"Exporting model to {export_format} format")
            
            model = YOLO(str(model_path))
            
            exported_path = model.export(
                format=export_format,
                device=YOLO_CONFIG['device']
            )
            
            self.logger.info(f"Model exported to: {exported_path}")
            return Path(exported_path)
            
        except Exception as e:
            self.logger.error(f"Error exporting model: {str(e)}")
            return None
    
    def resume_training(self, checkpoint_path: Path) -> Optional[Path]:
        """
        Resume training from a checkpoint
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Path to the trained model file
        """
        try:
            self.logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
            
            # Load model from checkpoint
            model = YOLO(str(checkpoint_path))
            
            # Resume training
            results = model.train(resume=True)
            
            # Get the final model path
            final_model_path = checkpoint_path.parent / 'weights' / 'best.pt'
            
            self.logger.info("Training resumed and completed successfully")
            return final_model_path
            
        except Exception as e:
            self.logger.error(f"Error resuming training: {str(e)}")
            return None 