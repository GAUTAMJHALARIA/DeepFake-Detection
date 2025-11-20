#!/bin/bash
# Script to download models during Docker build
# Set MODEL_DOWNLOAD_URL as build argument

set -e

MODEL_DIR="/models/deepfake"
MODEL_URL="${MODEL_DOWNLOAD_URL:-}"

if [ -z "$MODEL_URL" ]; then
    echo "Warning: MODEL_DOWNLOAD_URL not set. Models will not be downloaded."
    echo "If models are in build context, they will be copied instead."
    exit 0
fi

echo "Downloading models from: $MODEL_URL"

# Create models directory
mkdir -p "$MODEL_DIR"

# Download and extract
if [[ "$MODEL_URL" == *.zip ]]; then
    echo "Downloading zip file..."
    wget -q -O /tmp/models.zip "$MODEL_URL" || curl -L -o /tmp/models.zip "$MODEL_URL"
    echo "Extracting models..."
    unzip -q /tmp/models.zip -d "$MODEL_DIR" || unzip -q /tmp/models.zip -d /tmp && mv /tmp/models/deepfake/* "$MODEL_DIR/" || mv /tmp/deepfake/* "$MODEL_DIR/"
    rm -f /tmp/models.zip
elif [[ "$MODEL_URL" == *.tar.gz ]] || [[ "$MODEL_URL" == *.tgz ]]; then
    echo "Downloading tar.gz file..."
    wget -q -O /tmp/models.tar.gz "$MODEL_URL" || curl -L -o /tmp/models.tar.gz "$MODEL_URL"
    echo "Extracting models..."
    tar -xzf /tmp/models.tar.gz -C "$MODEL_DIR" --strip-components=1 || tar -xzf /tmp/models.tar.gz -C /tmp && mv /tmp/models/deepfake/* "$MODEL_DIR/" || mv /tmp/deepfake/* "$MODEL_DIR/"
    rm -f /tmp/models.tar.gz
else
    echo "Error: Unsupported file format. Use .zip or .tar.gz"
    exit 1
fi

# Verify models were extracted
if [ -d "$MODEL_DIR/1" ] || [ -f "$MODEL_DIR/1/saved_model.pb" ]; then
    echo "Models downloaded successfully!"
    ls -la "$MODEL_DIR"
else
    echo "Warning: Models may not be in expected structure. Checking contents..."
    find "$MODEL_DIR" -type f | head -10
fi
