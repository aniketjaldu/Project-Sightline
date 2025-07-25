@echo off
REM Project Sightline Workflow Runner - Windows Batch Script
REM This script provides both interactive and command-line access to the workflow

setlocal enabledelayedexpansion

REM Check for command line arguments - if provided, run in non-interactive mode
if "%~1" neq "" (
    goto batch_mode
)

echo ====================================
echo  Project Sightline Workflow Runner
echo ====================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies if requirements.txt exists
if exist "requirements.txt" (
    echo Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Warning: Some dependencies may not have installed correctly
    )
)

REM Check if .env file exists
if not exist ".env" (
    echo.
    echo Warning: .env file not found!
    echo Please copy env_example.txt to .env and configure your credentials.
    echo Required: LABELBOX_API_KEY, LABELBOX_PROJECT_ID
    echo Optional: YOLO_DEVICE, YOLO_BATCH_SIZE, YOLO_EPOCHS, YOLO_IMG_SIZE, YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD, YOLO_PRETRAINED_MODEL, YOLO_MAX_DET
    echo See README.md and env_example.txt for details.
    echo.
    if exist "env_example.txt" (
        echo You can use this command: copy env_example.txt .env
    )
    echo.
    pause
)

goto interactive_mode

:batch_mode
REM Non-interactive mode - pass all arguments to Python script
call venv\Scripts\activate.bat
python core/run_workflow.py %*
exit /b %errorlevel%

:interactive_mode
echo.
echo Choose workflow mode:
echo 1. Full Pipeline (Training + Inference)
echo 2. Training Only
echo 3. Inference Only
echo 4. Show Status
echo 5. Custom Command
echo 6. Help / Usage Examples
echo 7. Exit
echo 8. Compare Tracking Methods
echo.

set /p choice="Enter your choice (1-8): "

if "%choice%"=="1" goto full_pipeline
if "%choice%"=="2" goto training_only
if "%choice%"=="3" goto inference_only
if "%choice%"=="4" goto show_status
if "%choice%"=="5" goto custom_command
if "%choice%"=="6" goto show_help
if "%choice%"=="7" goto exit
if "%choice%"=="8" goto compare_mode
goto invalid_choice

:full_pipeline
echo.
echo === Full Pipeline Mode ===
echo This will run training followed by inference on multiple videos
echo.
set /p training_id="Enter training data row ID: "
if "!training_id!"=="" (
    echo Error: Training ID is required
    goto end
)

set /p inference_ids="Enter inference data row IDs (space-separated): "
if "!inference_ids!"=="" (
    echo Error: At least one inference ID is required
    goto end
)

set /p model_name="Enter model name: "
if "!model_name!"=="" (
    echo Error: Model name is required
    goto end
)

set /p no_import="Skip importing to Labelbox? (y/N): "
set /p workflow_id="Enter custom workflow ID (optional): "

set import_flag=
if /i "!no_import!"=="y" set import_flag=--no-import

set workflow_flag=
if "!workflow_id!" neq "" set workflow_flag=--workflow-id "!workflow_id!"

set /p tracking_method="Enter tracking method (sightline/bytetrack/botsort, default: sightline): "
set tracking_flag=
if not "!tracking_method!"=="" set tracking_flag=--tracking-method !tracking_method!

echo.
echo Running: python core/run_workflow.py --mode full --training-id "!training_id!" --inference-ids !inference_ids! --model-name "!model_name!" !import_flag! !workflow_flag! !tracking_flag!
echo.

python core/run_workflow.py --mode full --training-id "!training_id!" --inference-ids !inference_ids! --model-name "!model_name!" !import_flag! !workflow_flag! !tracking_flag!
goto end

:training_only
echo.
echo === Training Only Mode ===
echo This will download data, train a model, and clean up training data
echo.
set /p training_id="Enter training data row ID: "
if "!training_id!"=="" (
    echo Error: Training ID is required
    goto end
)

set /p model_name="Enter model name: "
if "!model_name!"=="" (
    echo Error: Model name is required
    goto end
)

set /p workflow_id="Enter custom workflow ID (optional): "

set workflow_flag=
if "!workflow_id!" neq "" set workflow_flag=--workflow-id "!workflow_id!"

echo.
echo Running: python core/run_workflow.py --mode training --training-id "!training_id!" --model-name "!model_name!" !workflow_flag!
echo.

python core/run_workflow.py --mode training --training-id "!training_id!" --model-name "!model_name!" !workflow_flag!
goto end

:inference_only
echo.
echo === Inference Only Mode ===
echo This will run inference on videos using an existing model
echo.
set /p inference_ids="Enter inference data row IDs (space-separated): "
if "!inference_ids!"=="" (
    echo Error: At least one inference ID is required
    goto end
)

echo.
echo Choose model selection method:
echo 1. Use model by name (searches multiple locations)
echo 2. Use model by file path
echo.
set /p model_choice="Enter choice (1-2): "

set model_flag=
if "!model_choice!"=="1" (
    set /p model_name="Enter model name: "
    if "!model_name!"=="" (
        echo Error: Model name is required
        goto end
    )
    set model_flag=--model-name "!model_name!"
) else if "!model_choice!"=="2" (
    set /p model_path="Enter model path: "
    if "!model_path!"=="" (
        echo Error: Model path is required
        goto end
    )
    set model_flag=--model-path "!model_path!"
) else (
    echo Error: Invalid choice
    goto end
)

set /p no_import="Skip importing to Labelbox? (y/N): "
set /p workflow_id="Enter custom workflow ID (optional): "

set import_flag=
if /i "!no_import!"=="y" set import_flag=--no-import

set workflow_flag=
if "!workflow_id!" neq "" set workflow_flag=--workflow-id "!workflow_id!"

set /p tracking_method="Enter tracking method (sightline/bytetrack/botsort, default: sightline): "
set tracking_flag=
if not "!tracking_method!"=="" set tracking_flag=--tracking-method !tracking_method!

echo.
echo Running: python core/run_workflow.py --mode inference --inference-ids !inference_ids! !model_flag! !import_flag! !workflow_flag! !tracking_flag!
echo.

python core/run_workflow.py --mode inference --inference-ids !inference_ids! !model_flag! !import_flag! !workflow_flag! !tracking_flag!
goto end

:show_status
echo.
echo === Workflow Status ===
python core/run_workflow.py --mode status
goto end

:custom_command
echo.
echo === Custom Command ===
echo Available options:
echo   --mode: full, training, inference, status
echo   --training-id: Data row ID for training
echo   --inference-ids: Space-separated inference data row IDs
echo   --model-name: Model name (for training or inference)
echo   --model-path: Path to model file (for inference)
echo   --no-import: Skip importing annotations to Labelbox
echo   --workflow-id: Custom workflow ID
echo.
echo Example: --mode full --training-id "abc123" --inference-ids "def456" "ghi789" --model-name "my_model" --no-import
echo.
set /p custom_cmd="Enter your command arguments: "
if "!custom_cmd!"=="" (
    echo Error: No command provided
    goto end
)

echo.
echo Running: python core/run_workflow.py !custom_cmd!
echo.

python core/run_workflow.py !custom_cmd!
goto end

:compare_mode
echo.
echo === Compare Tracking Methods Mode ===
echo This will compare tracking methods on inference videos using an existing model
echo.
set /p inference_ids="Enter inference data row IDs (space-separated): "
if "!inference_ids!"=="" (
    echo Error: At least one inference ID is required
    goto end
)
set /p model_name="Enter model name (leave blank to use model path): "
set model_flag=
if not "!model_name!"=="" (
    set model_flag=--model-name "!model_name!"
) else (
    set /p model_path="Enter model path: "
    if "!model_path!"=="" (
        echo Error: Model path is required
        goto end
    )
    set model_flag=--model-path "!model_path!"
)
set /p output_dir="Enter output directory for comparison results: "
if "!output_dir!"=="" (
    echo Error: Output directory is required
    goto end
)
set /p comparison_methods="Enter comparison methods (space-separated, default: sightline bytetrack botsort): "
set comparison_flag=
if not "!comparison_methods!"=="" (
    set comparison_flag=--comparison-methods !comparison_methods!
)
set /p workflow_id="Enter custom workflow ID (optional): "
set workflow_flag=
if not "!workflow_id!"=="" set workflow_flag=--workflow-id "!workflow_id!"

echo.
echo Running: python core/run_workflow.py --mode compare --inference-ids !inference_ids! !model_flag! --output-dir "!output_dir!" !comparison_flag! !workflow_flag!
echo.
python core/run_workflow.py --mode compare --inference-ids !inference_ids! !model_flag! --output-dir "!output_dir!" !comparison_flag! !workflow_flag!
goto end

:show_help
echo.
echo === Usage Examples ===
echo.
echo 1. Full Pipeline:
echo    run_workflow.bat --mode full --training-id "train123" --inference-ids "inf1" "inf2" --model-name "my_model"
echo.
echo 2. Training Only:
echo    run_workflow.bat --mode training --training-id "train123" --model-name "my_model"
echo.
echo 3. Inference Only (by model name):
echo    run_workflow.bat --mode inference --inference-ids "inf1" "inf2" --model-name "my_model"
echo.
echo 4. Inference Only (by model path):
echo    run_workflow.bat --mode inference --inference-ids "inf1" "inf2" --model-path "path/to/model.pt"
echo.
echo 5. Skip Labelbox Import:
echo    run_workflow.bat --mode inference --inference-ids "inf1" --model-name "my_model" --no-import
echo.
echo 6. Custom Workflow ID:
echo    run_workflow.bat --mode full --training-id "train123" --inference-ids "inf1" --model-name "my_model" --workflow-id "custom_id"
echo.
echo 7. Show Status:
echo    run_workflow.bat --mode status
echo.
echo 8. Compare Tracking Methods:
echo    run_workflow.bat --mode compare --inference-ids "inf1" "inf2" --model-name "my_model" --output-dir "output_dir" --comparison-methods "sightline bytetrack botsort"
echo.
echo Note: You can run this script with command line arguments for batch processing,
echo       or run it without arguments for interactive mode.
echo.
goto end

:invalid_choice
echo Invalid choice. Please try again.
goto end

:end
echo.
if "%~1"=="" (
    pause
)

:exit
if "%~1"=="" (
    echo Goodbye!
)
exit /b 0 