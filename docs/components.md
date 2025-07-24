# Core Components Documentation

## Software Components

- **core/config.py**: Loads environment variables, sets up directory structure, and provides global configuration for the pipeline.
- **core/main_workflow.py**: Orchestrates the main workflow, including data fetching, processing, training, and inference.
- **core/model_trainer.py**: Handles YOLO model training, including transfer learning and saving model artifacts.
- **core/run_workflow.py**: Command-line entry point for running the workflow in different modes (training, inference, full, status).
- **pipeline/data_fetcher.py**: Downloads data and annotations from Labelbox.
- **pipeline/data_processor.py**: Processes raw data and converts it to YOLO format.
- **pipeline/inference_runner.py**: Runs inference on videos using trained YOLO models.
- **pipeline/enhanced_inference_runner.py**: Provides advanced inference and tracking options.
- **pipeline/annotation_importer.py**: Imports model predictions back into Labelbox as annotations.
- **pipeline/tracking_manager.py**: Manages different tracking methods (Sightline, ByteTrack, BoT-SORT).
- **pipeline/delete_prelabels.py**: Utility for deleting prelabels and regular labels from Labelbox.
- **utils/logger.py**: Sets up logging for all components.
- **utils/workflow_manager.py**: Manages workflow directory creation and manifest files.

## Configuration Files
- **env_example.txt**: Template for all required and optional environment variables.
- **project_data/configs/trackers/valorant_bytetrack.yaml**: Custom ByteTrack config for Valorant.
- **project_data/configs/trackers/valorant_botsort.yaml**: Custom BoT-SORT config for Valorant.
- **project_data/configs/datasets/**: Dataset configuration templates.
- **project_data/configs/models/**: Model configuration templates.

## Hardware Requirements
- **GPU:** Strongly recommended for training and inference (NVIDIA GPU with CUDA support preferred).
- **CPU:** Supported for small-scale testing or CPU-only runs (set `YOLO_DEVICE=cpu`).
- **RAM:** At least 8GB recommended for training and inference.

## Network Requirements
- **Internet Access:** Required for downloading data from Labelbox and for initial setup (dependency installation).
- **Labelbox Account:** Required for annotation management and API access.

## Database/Storage
- No external database is used. All data is stored in the organized directory structure under `project_data/`. 