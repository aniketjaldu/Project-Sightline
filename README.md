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

You can use the provided scripts for both interactive and batch operation:

- **Windows:** Run `run_workflow.bat` (double-click or from command line)
- **Unix/Mac/Linux:** Run `./run_workflow.sh`

Both scripts:
- Set up the environment and dependencies
- Guide you through interactive options or accept command-line arguments

See the script or use the `Help / Usage Examples` menu for command-line options.

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