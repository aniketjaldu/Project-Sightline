# Project Sightline

Project Sightline is an end-to-end pipeline for training and running object detection and tracking models on Valorant gameplay, using Labelbox for annotation management and YOLO for model training/inference. The project is designed for reproducibility, modularity, and ease of use for both research and production workflows.

---

## Quickstart

1. **Clone the repository** and navigate to the project root.
2. **Python 3.8+ is required.**
3. **Create a virtual environment:**
   - Windows: `python -m venv venv`
   - Unix/Mac: `python3 -m venv venv`
4. **Activate the virtual environment:**
   - Windows: `venv\Scripts\activate.bat`
   - Unix/Mac: `source venv/bin/activate`
5. **Install dependencies:**
   - `pip install -r requirements.txt`
6. **Configure environment variables:**
   - Copy `env_example.txt` to `.env` and fill in your Labelbox credentials and any custom YOLO settings.

---

## Directory Structure

```
project_data/                  # Root data directory
├── workflows/                 # Individual workflow runs (training/inference)
├── models/                    # Permanent model storage
├── configs/                   # Configuration files (trackers, datasets, models)
├── temp/                      # Temporary files (downloads, processing, inference)
├── logs/                      # Centralized logging
├── archive/                   # Completed/archived workflows

core/                          # Main workflow, configuration, and model training logic
pipeline/                      # Data fetching, annotation import/export, tracking, and inference
utils/                         # Logging and workflow management utilities
docs/                          # All design documents, notes, and non-code documentation
```

---

## Environment Variables

See `env_example.txt` for all options. **Required:**
- `LABELBOX_API_KEY` (get from https://app.labelbox.com/account/api-keys)
- `LABELBOX_PROJECT_ID` (from your Labelbox project URL)

Optional YOLO overrides:
- `YOLO_DEVICE`, `YOLO_BATCH_SIZE`, `YOLO_EPOCHS`, `YOLO_IMG_SIZE`, `YOLO_CONF_THRESHOLD`, `YOLO_IOU_THRESHOLD`, `YOLO_PRETRAINED_MODEL`, `YOLO_MAX_DET`

---

## Running the Workflow

Run the workflow directly using Python's module mode:

```
python -m core.run_workflow [arguments]
```

This command works on all platforms (Windows, Mac, Linux) and supports both interactive and batch operation.

**Examples:**

- Full pipeline (training + inference):
  ```
  python -m core.run_workflow --mode full --training-id TRAIN_ID --inference-ids INF_ID1 INF_ID2 --model-name my_model
  ```
- Training only:
  ```
  python -m core.run_workflow --mode training --training-id TRAIN_ID --model-name my_model
  ```
- Inference only:
  ```
  python -m core.run_workflow --mode inference --inference-ids INF_ID1 INF_ID2 --model-path path/to/model.pt
  ```
- Compare tracking methods:
  ```
  python -m core.run_workflow --mode compare --inference-ids INF_ID1 INF_ID2 --model-name my_model --output-dir output_dir --comparison-methods sightline bytetrack botsort
  ```

For all available options, run:
```
python -m core.run_workflow --help
```

---

## Command-Line Parameters

The following parameters are available for `python -m core.run_workflow`:

| Parameter              | Type    | Choices                                 | Default     | Required For                | Description |
|------------------------|---------|-----------------------------------------|-------------|-----------------------------|-------------|
| `--mode`               | str     | training, inference, full, status, compare | —           | all                         | Workflow mode to run |
| `--training-id`        | str     | —                                       | —           | training, full              | Data row ID for training data |
| `--inference-ids`      | str+    | —                                       | —           | inference, full, compare    | Data row IDs for inference videos (space-separated) |
| `--model-name`         | str     | —                                       | —           | training, inference, full, compare | Name for the trained model (for training) or existing model name (for inference/compare) |
| `--model-path`         | str     | —                                       | —           | inference, compare (if not using --model-name) | Path to existing trained model |
| `--no-import`          | flag    | —                                       | false       | any                         | Skip importing annotations back to Labelbox |
| `--tracking-method`    | str     | sightline, bytetrack, botsort           | sightline   | inference, full             | Tracking method to use |
| `--comparison-methods` | str+    | sightline, bytetrack, botsort           | all         | compare                     | Tracking methods to compare (space-separated) |
| `--output-dir`         | str     | —                                       | —           | compare                     | Output directory for comparison results |
| `--workflow-id`        | str     | —                                       | —           | optional                    | Custom workflow ID |

**Notes:**
- For `--inference-ids` and `--comparison-methods`, provide space-separated values.
- For `--no-import`, just include the flag (no value needed) to skip importing results to Labelbox.
- For a full list and help, run:
  ```
  python -m core.run_workflow --help
  ```

---

## Documentation & Design

All design documents, project notes, technical documentation, and progress reports are located in the [`docs/`](./docs/) directory. This includes:
- Final design document
- Project goals and status
- Notes on interesting, weird, or unexpected items
- Documentation of all core components and configuration
- Versioned documentation of scripts, configs, and any design diagrams
- All other non-code documents and notes

For detailed information, please refer to the files in [`docs/`](./docs/).

---

## Support
For any issues, check the logs in `project_data/logs/`. 