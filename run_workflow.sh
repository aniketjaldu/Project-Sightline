#!/bin/bash
# Project Sightline Workflow Runner - Unix/Linux Shell Script
# This script provides both interactive and command-line access to the workflow

set -e  # Exit on any error

# Check for command line arguments - if provided, run in non-interactive mode
if [ $# -gt 0 ]; then
    # Activate virtual environment and run Python script with all arguments
    source venv/bin/activate 2>/dev/null || true
    python3 core/run_workflow.py "$@"
    exit $?
fi

echo "===================================="
echo "  Project Sightline Workflow Runner"
echo "===================================="
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Warning: Some dependencies may not have installed correctly"
    fi
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo
    echo "Warning: .env file not found!"
    echo "Please copy env_example.txt to .env and configure your credentials."
    echo "Required: LABELBOX_API_KEY, LABELBOX_PROJECT_ID"
    echo "Optional: YOLO_DEVICE, YOLO_BATCH_SIZE, YOLO_EPOCHS, YOLO_IMG_SIZE, YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD, YOLO_PRETRAINED_MODEL, YOLO_MAX_DET"
    echo "See README.md and env_example.txt for details."
    echo
    if [ -f "env_example.txt" ]; then
        echo "You can use this command: cp env_example.txt .env"
    fi
    echo
    read -p "Press Enter to continue..."
fi

# Function to validate required input
validate_input() {
    local input="$1"
    local field_name="$2"
    
    if [ -z "$input" ]; then
        echo "Error: $field_name is required"
        return 1
    fi
    return 0
}

# Function to run full pipeline
run_full_pipeline() {
    echo
    echo "=== Full Pipeline Mode ==="
    echo "This will run training followed by inference on multiple videos"
    echo
    
    read -p "Enter training data row ID: " training_id
    validate_input "$training_id" "Training ID" || return 1
    
    read -p "Enter inference data row IDs (space-separated): " inference_ids
    validate_input "$inference_ids" "Inference IDs" || return 1
    
    read -p "Enter model name: " model_name
    validate_input "$model_name" "Model name" || return 1
    
    read -p "Skip importing to Labelbox? (y/N): " no_import
    read -p "Enter custom workflow ID (optional): " workflow_id
    
    # Build command
    cmd="python3 core/run_workflow.py --mode full --training-id \"$training_id\" --inference-ids $inference_ids --model-name \"$model_name\""
    
    if [ "$no_import" = "y" ] || [ "$no_import" = "Y" ]; then
        cmd="$cmd --no-import"
    fi
    
    if [ -n "$workflow_id" ]; then
        cmd="$cmd --workflow-id \"$workflow_id\""
    fi
    
    echo
    echo "Running: $cmd"
    echo
    
    eval $cmd
}

# Function to run training only
run_training_only() {
    echo
    echo "=== Training Only Mode ==="
    echo "This will download data, train a model, and clean up training data"
    echo
    
    read -p "Enter training data row ID: " training_id
    validate_input "$training_id" "Training ID" || return 1
    
    read -p "Enter model name: " model_name
    validate_input "$model_name" "Model name" || return 1
    
    read -p "Enter custom workflow ID (optional): " workflow_id
    
    # Build command
    cmd="python3 core/run_workflow.py --mode training --training-id \"$training_id\" --model-name \"$model_name\""
    
    if [ -n "$workflow_id" ]; then
        cmd="$cmd --workflow-id \"$workflow_id\""
    fi
    
    echo
    echo "Running: $cmd"
    echo
    
    eval $cmd
}

# Function to run inference only
run_inference_only() {
    echo
    echo "=== Inference Only Mode ==="
    echo "This will run inference on videos using an existing model"
    echo
    
    read -p "Enter inference data row IDs (space-separated): " inference_ids
    validate_input "$inference_ids" "Inference IDs" || return 1
    
    echo
    echo "Choose model selection method:"
    echo "1. Use model by name (searches multiple locations)"
    echo "2. Use model by file path"
    echo
    read -p "Enter choice (1-2): " model_choice
    
    local model_flag=""
    case $model_choice in
        1)
            read -p "Enter model name: " model_name
            validate_input "$model_name" "Model name" || return 1
            model_flag="--model-name \"$model_name\""
            ;;
        2)
            read -p "Enter model path: " model_path
            validate_input "$model_path" "Model path" || return 1
            model_flag="--model-path \"$model_path\""
            ;;
        *)
            echo "Error: Invalid choice"
            return 1
            ;;
    esac
    
    read -p "Skip importing to Labelbox? (y/N): " no_import
    read -p "Enter custom workflow ID (optional): " workflow_id
    
    # Add tracking-method prompt
    read -p "Enter tracking method (sightline/bytetrack/botsort, default: sightline): " tracking_method
    if [ -n "$tracking_method" ]; then
        tracking_flag="--tracking-method $tracking_method"
    else
        tracking_flag=""
    fi

    # Build command
    cmd="python3 core/run_workflow.py --mode inference --inference-ids $inference_ids $model_flag $tracking_flag"
    
    if [ "$no_import" = "y" ] || [ "$no_import" = "Y" ]; then
        cmd="$cmd --no-import"
    fi
    
    if [ -n "$workflow_id" ]; then
        cmd="$cmd --workflow-id \"$workflow_id\""
    fi
    
    echo
    echo "Running: $cmd"
    echo
    
    eval $cmd
}

# Function to show status
show_status() {
    echo
    echo "=== Workflow Status ==="
    python3 core/run_workflow.py --mode status
}

# Function to run custom command
run_custom_command() {
    echo
    echo "=== Custom Command ==="
    echo "Available options:"
    echo "  --mode: full, training, inference, status"
    echo "  --training-id: Data row ID for training"
    echo "  --inference-ids: Space-separated inference data row IDs"
    echo "  --model-name: Model name (for training or inference)"
    echo "  --model-path: Path to model file (for inference)"
    echo "  --no-import: Skip importing annotations to Labelbox"
    echo "  --workflow-id: Custom workflow ID"
    echo
    echo "Example: --mode full --training-id \"abc123\" --inference-ids \"def456\" \"ghi789\" --model-name \"my_model\" --no-import"
    echo
    read -p "Enter your command arguments: " custom_cmd
    
    if [ -z "$custom_cmd" ]; then
        echo "Error: No command provided"
        return 1
    fi
    
    echo
    echo "Running: python3 core/run_workflow.py $custom_cmd"
    echo
    
    python3 core/run_workflow.py $custom_cmd
}

# Function to run compare mode
run_compare_mode() {
    echo
    echo "=== Compare Tracking Methods Mode ==="
    echo "This will compare tracking methods on inference videos using an existing model"
    echo
    read -p "Enter inference data row IDs (space-separated): " inference_ids
    validate_input "$inference_ids" "Inference IDs" || return 1
    read -p "Enter model name (leave blank to use model path): " model_name
    model_flag=""
    if [ -n "$model_name" ]; then
        model_flag="--model-name \"$model_name\""
    else
        read -p "Enter model path: " model_path
        validate_input "$model_path" "Model path" || return 1
        model_flag="--model-path \"$model_path\""
    fi
    read -p "Enter output directory for comparison results: " output_dir
    validate_input "$output_dir" "Output directory" || return 1
    read -p "Enter comparison methods (space-separated, default: sightline bytetrack botsort): " comparison_methods
    if [ -n "$comparison_methods" ]; then
        comparison_flag="--comparison-methods $comparison_methods"
    else
        comparison_flag=""
    fi
    read -p "Enter custom workflow ID (optional): " workflow_id
    if [ -n "$workflow_id" ]; then
        workflow_flag="--workflow-id \"$workflow_id\""
    else
        workflow_flag=""
    fi
    cmd="python3 core/run_workflow.py --mode compare --inference-ids $inference_ids $model_flag --output-dir \"$output_dir\" $comparison_flag $workflow_flag"
    echo
    echo "Running: $cmd"
    echo
    eval $cmd
}

# Function to show help
show_help() {
    echo
    echo "=== Usage Examples ==="
    echo
    echo "1. Full Pipeline:"
    echo "   ./run_workflow.sh --mode full --training-id \"train123\" --inference-ids \"inf1\" \"inf2\" --model-name \"my_model\""
    echo
    echo "2. Training Only:"
    echo "   ./run_workflow.sh --mode training --training-id \"train123\" --model-name \"my_model\""
    echo
    echo "3. Inference Only (by model name):"
    echo "   ./run_workflow.sh --mode inference --inference-ids \"inf1\" \"inf2\" --model-name \"my_model\""
    echo
    echo "4. Inference Only (by model path):"
    echo "   ./run_workflow.sh --mode inference --inference-ids \"inf1\" \"inf2\" --model-path \"path/to/model.pt\""
    echo
    echo "5. Skip Labelbox Import:"
    echo "   ./run_workflow.sh --mode inference --inference-ids \"inf1\" --model-name \"my_model\" --no-import"
    echo
    echo "6. Custom Workflow ID:"
    echo "   ./run_workflow.sh --mode full --training-id \"train123\" --inference-ids \"inf1\" --model-name \"my_model\" --workflow-id \"custom_id\""
    echo
    echo "7. Show Status:"
    echo "   ./run_workflow.sh --mode status"
    echo
    echo "8. Compare Tracking Methods:"
    echo "   ./run_workflow.sh --mode compare --inference-ids \"inf1\" \"inf2\" --model-name \"my_model\" --output-dir \"/path/to/output\" --comparison-methods \"sightline bytetrack botsort\""
    echo
    echo "Note: You can run this script with command line arguments for batch processing,"
    echo "      or run it without arguments for interactive mode."
    echo
}

# Main interactive menu
while true; do
    echo
    echo "Choose workflow mode:"
    echo "1. Full Pipeline (Training + Inference)"
    echo "2. Training Only"
    echo "3. Inference Only"
    echo "4. Show Status"
    echo "5. Custom Command"
    echo "6. Help / Usage Examples"
    echo "7. Exit"
    echo
    
    read -p "Enter your choice (1-7): " choice
    
    case $choice in
        1)
            run_full_pipeline
            ;;
        2)
            run_training_only
            ;;
        3)
            run_inference_only
            ;;
        4)
            show_status
            ;;
        5)
            run_custom_command
            ;;
        6)
            show_help
            ;;
        7)
            echo "Goodbye!"
            exit 0
            ;;
        8)
            run_compare_mode
            ;;
        *)
            echo "Invalid choice. Please try again."
            ;;
    esac
    
    echo
    read -p "Press Enter to continue..."
done 