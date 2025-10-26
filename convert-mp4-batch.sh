#!/bin/bash

echo "MP4 Batch Converter for Deepfake Detection System"
echo "================================================"
echo ""
echo "This script converts all MP4 files in the current folder to browser-compatible H.264 format."
echo ""
echo "Requirements:"
echo "- FFmpeg must be installed"
echo "- Run this script in the folder containing your MP4 files"
echo ""
echo "Press Enter to continue or Ctrl+C to cancel..."
read

echo ""
echo "Starting conversion..."
echo ""

count=0
for file in *.mp4; do
    if [ -f "$file" ]; then
        count=$((count + 1))
        echo "Converting $file..."
        ffmpeg -i "$file" -c:v libx264 -profile:v baseline -c:a aac -y "converted_$file"
        if [ $? -eq 0 ]; then
            echo "✓ Successfully converted $file"
        else
            echo "✗ Failed to convert $file"
        fi
        echo ""
    fi
done

echo ""
echo "Conversion complete!"
echo "Converted $count files."
echo ""
echo "Original files are preserved. Converted files have 'converted_' prefix."
echo ""
read -p "Press Enter to exit..."
