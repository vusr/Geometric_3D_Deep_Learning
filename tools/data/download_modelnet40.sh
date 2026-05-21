#!/bin/bash
# Download and extract raw ModelNet40 OFF mesh files
# Output: ~/geometric_deep_learning/data/raw/ModelNet40/<class>/{train,test}/*.off
# Size: ~450 MB compressed, ~1.8 GB extracted

set -e

PROJECT_DIR="$HOME/geometric_deep_learning"
DATA_DIR="$PROJECT_DIR/data"
RAW_DIR="$DATA_DIR/raw"
ZIP_PATH="$DATA_DIR/ModelNet40.zip"

# Primary URL (Princeton/ShapeNet CDN)
PRIMARY_URL="http://modelnet.cs.princeton.edu/ModelNet40.zip"
# Fallback mirror
FALLBACK_URL="https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip"

mkdir -p "$RAW_DIR"

echo "=== Downloading ModelNet40 raw OFF files ==="
if [ -d "$RAW_DIR/ModelNet40" ] && [ "$(ls -A $RAW_DIR/ModelNet40)" ]; then
    echo "ModelNet40 already present at $RAW_DIR/ModelNet40, skipping download."
else
    if [ ! -f "$ZIP_PATH" ]; then
        echo "Trying primary URL: $PRIMARY_URL"
        if ! wget -q --show-progress -O "$ZIP_PATH" "$PRIMARY_URL"; then
            echo "Primary URL failed. Trying fallback..."
            rm -f "$ZIP_PATH"
            wget -q --show-progress -O "$ZIP_PATH" "$FALLBACK_URL"
        fi
        echo "Download complete."
    else
        echo "Zip file already exists at $ZIP_PATH, skipping download."
    fi

    echo "Extracting to $RAW_DIR ..."
    unzip -q "$ZIP_PATH" -d "$RAW_DIR"

    # The zip extracts as ModelNet40/ folder containing class subfolders
    if [ ! -d "$RAW_DIR/ModelNet40" ]; then
        # Sometimes the top-level folder is named differently
        EXTRACTED=$(ls "$RAW_DIR" | head -1)
        mv "$RAW_DIR/$EXTRACTED" "$RAW_DIR/ModelNet40"
    fi

    echo "Extraction complete."
    echo "Removing zip file to save space..."
    rm -f "$ZIP_PATH"
fi

echo ""
echo "=== Dataset Statistics ==="
echo "Classes found: $(ls $RAW_DIR/ModelNet40 | wc -l)"
echo "Train shapes : $(find $RAW_DIR/ModelNet40 -path '*/train/*.off' | wc -l)"
echo "Test shapes  : $(find $RAW_DIR/ModelNet40 -path '*/test/*.off' | wc -l)"
echo ""
echo "Data ready at: $RAW_DIR/ModelNet40"
