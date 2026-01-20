# #!/bin/bash

# # ==============================================================================
# # Script: Batch Table Extraction from Images
# # Description: Extract tables from multiple images and convert to HTML
# # ==============================================================================

# # Default configuration
# DEFAULT_INPUT_DIR="/home/jovyan/nas_comm/workspace/1_user/anhdungitvn@agilesoda.ai/113/anhdungitvn@agilesoda.ai/TRivia/data/evaluation_filter_notnt/images/val"
# DEFAULT_OUTPUT_DIR="outputs"
# BACKEND="tsr"

# # Parse command line arguments
# INPUT_DIR="${1:-$DEFAULT_INPUT_DIR}"
# OUTPUT_DIR="${2:-$DEFAULT_OUTPUT_DIR}"

# # Validate input directory
# if [ ! -d "$INPUT_DIR" ]; then
#     echo "Error: Input directory not found: $INPUT_DIR"
#     echo "Usage: $0 [input_dir] [output_dir]"
#     exit 1
# fi

# # Create output directory if it doesn't exist
# mkdir -p "$OUTPUT_DIR"

# # Counter for statistics
# total_files=0
# success_count=0
# failed_count=0

# echo "=================================================="
# echo "Batch Table Extraction"
# echo "=================================================="
# echo "Input directory : $INPUT_DIR"
# echo "Output directory: $OUTPUT_DIR"
# echo "Backend         : $BACKEND"
# echo "=================================================="
# echo ""

# # Find all image files (handle special characters in filenames)
# image_files=()
# while IFS= read -r -d '' file; do
#     image_files+=("$file")
# done < <(find "$INPUT_DIR" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) -print0 | sort -z)

# total_files=${#image_files[@]}

# if [ $total_files -eq 0 ]; then
#     echo "No image files found in $INPUT_DIR"
#     exit 1
# fi

# echo "Found $total_files image file(s)"
# echo ""

# # Process each image file
# for img_path in "${image_files[@]}"; do
#     # Get filename without extension
#     filename=$(basename "$img_path")
#     name_no_ext="${filename%.*}"
    
#     # Output HTML path
#     output_html="$OUTPUT_DIR/${name_no_ext}.html"
    
#     echo "[$((success_count + failed_count + 1))/$total_files] Processing: $filename"
    
#     # Run the agent
#     if python agent/run_agent.py --input "Extract table from $img_path using backend='$BACKEND'. Convert to HTML and save to $output_html." 2>&1 | tee -a "$OUTPUT_DIR/process.log"; then
#         if [ -f "$output_html" ]; then
#             echo "  ✓ Success: $output_html"
#             ((success_count++))
#         else
#             echo "  ✗ Failed: Output file not created"
#             ((failed_count++))
#         fi
#     else
#         echo "  ✗ Failed: Command execution error"
#         ((failed_count++))
#     fi
#     echo ""
# done

# # Print summary
# echo "=================================================="
# echo "Summary"
# echo "=================================================="
# echo "Total files    : $total_files"
# echo "Success        : $success_count"
# echo "Failed         : $failed_count"
# echo "Success rate   : $(awk "BEGIN {printf \"%.2f\", ($success_count/$total_files)*100}")%"
# echo "Output dir     : $OUTPUT_DIR"
# echo "Log file       : $OUTPUT_DIR/process.log"
# echo "=================================================="

# # Exit with error if any failed
# if [ $failed_count -gt 0 ]; then
#     exit 1
# fi



#!/bin/bash

# ==============================================================================
# Script: Batch Table Extraction from Images (resume-able)
# Description: Extract tables from multiple images and convert to HTML
# Notes:
#   - If output HTML already exists (and non-empty), skip the image.
#   - Safe to run multiple times; only processes missing outputs.
# ==============================================================================

set -u  # error on unset variables

# Default configuration
DEFAULT_INPUT_DIR="/home/jovyan/nas_comm/workspace/1_user/anhdungitvn@agilesoda.ai/113/anhdungitvn@agilesoda.ai/TRivia/data/evaluation_filter_notnt/images/val"
DEFAULT_OUTPUT_DIR="outputs"
BACKEND="tsr"

# Parse command line arguments
INPUT_DIR="${1:-$DEFAULT_INPUT_DIR}"
OUTPUT_DIR="${2:-$DEFAULT_OUTPUT_DIR}"

# Validate input directory
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory not found: $INPUT_DIR"
    echo "Usage: $0 [input_dir] [output_dir]"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

LOG_FILE="$OUTPUT_DIR/process.log"

# Counter for statistics
total_files=0
planned_files=0
skipped_count=0
success_count=0
failed_count=0

echo "=================================================="
echo "Batch Table Extraction (resume-able)"
echo "=================================================="
echo "Input directory : $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Backend         : $BACKEND"
echo "Log file        : $LOG_FILE"
echo "=================================================="
echo ""

# Find all image files (handle special characters in filenames)
image_files=()
while IFS= read -r -d '' file; do
    image_files+=("$file")
done < <(find "$INPUT_DIR" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) -print0 | sort -z)

total_files=${#image_files[@]}

if [ $total_files -eq 0 ]; then
    echo "No image files found in $INPUT_DIR"
    exit 1
fi

echo "Found $total_files image file(s)"
echo ""

# Pre-scan: count how many will actually run (missing outputs)
for img_path in "${image_files[@]}"; do
    filename=$(basename "$img_path")
    name_no_ext="${filename%.*}"
    output_html="$OUTPUT_DIR/${name_no_ext}.html"

    # Skip if output exists and is non-empty
    if [ -s "$output_html" ]; then
        ((skipped_count++))
    else
        ((planned_files++))
    fi
done

echo "Will process     : $planned_files"
echo "Already done     : $skipped_count (skipped)"
echo ""

# Process each image file
idx=0
for img_path in "${image_files[@]}"; do
    ((idx++))

    filename=$(basename "$img_path")
    name_no_ext="${filename%.*}"
    output_html="$OUTPUT_DIR/${name_no_ext}.html"

    # Skip if output exists and is non-empty
    if [ -s "$output_html" ]; then
        echo "[$idx/$total_files] Skipping (exists): $filename -> $output_html"
        continue
    fi

    echo "[$idx/$total_files] Processing: $filename"
    echo "  Output: $output_html"

    # Run the agent
    if python agent/run_agent.py --input "Extract table from $img_path using backend='$BACKEND'. Convert to HTML and save to $output_html." 2>&1 | tee -a "$LOG_FILE"; then
        if [ -s "$output_html" ]; then
            echo "  ✓ Success: $output_html"
            ((success_count++))
        else
            echo "  ✗ Failed: Output file not created (or empty)"
            ((failed_count++))
        fi
    else
        echo "  ✗ Failed: Command execution error"
        ((failed_count++))
    fi
    echo ""
done

# Print summary
echo "=================================================="
echo "Summary"
echo "=================================================="
echo "Total images    : $total_files"
echo "Skipped (done)  : $skipped_count"
echo "Attempted       : $planned_files"
echo "Success         : $success_count"
echo "Failed          : $failed_count"
if [ $planned_files -gt 0 ]; then
    echo "Success rate    : $(awk "BEGIN {printf \"%.2f\", ($success_count/$planned_files)*100}")% (of attempted)"
else
    echo "Success rate    : N/A (nothing attempted)"
fi
echo "Output dir      : $OUTPUT_DIR"
echo "Log file        : $LOG_FILE"
echo "=================================================="

# Exit with error if any failed (same behavior style as before)
if [ $failed_count -gt 0 ]; then
    exit 1
fi
