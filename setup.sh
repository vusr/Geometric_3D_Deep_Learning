#!/bin/bash
# VM bootstrap script: run once after SSH into the GCP VM
# Installs Miniconda and creates the 'gdl' conda environment

set -e

PROJECT_DIR="$HOME/geometric_deep_learning"
CONDA_DIR="$HOME/miniconda3"
ENV_NAME="gdl"

echo "=== Step 1: Create project directory ==="
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

echo "=== Step 2: Install Miniconda (if not present) ==="
if [ ! -d "$CONDA_DIR" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm /tmp/miniconda.sh
    echo "Miniconda installed at $CONDA_DIR"
else
    echo "Miniconda already present at $CONDA_DIR"
fi

# Initialize conda in shell
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda init bash
source ~/.bashrc 2>/dev/null || true

echo "=== Step 3: Create/update conda environment '$ENV_NAME' ==="
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' already exists, updating..."
    conda env update -n "$ENV_NAME" -f environment.yml --prune
else
    # Install PyTorch-related PyG wheels separately (version-locked)
    conda create -n "$ENV_NAME" python=3.10 numpy pandas scikit-learn tqdm matplotlib -y

    conda activate "$ENV_NAME"

    # Install PyTorch with CUDA 12.1
    pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
        --index-url https://download.pytorch.org/whl/cu121

    # Install PyG and its dependencies (CUDA 12.1 / torch 2.3.0 wheels)
    pip install torch_geometric==2.5.3
    pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
        -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

    # Install Trimesh with all extras (mesh repair, etc.)
    pip install "trimesh[easy]==4.3.2" rtree open3d==0.18.0
fi

echo "=== Step 4: Verify GPU visibility ==="
conda activate "$ENV_NAME"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "=== Setup complete! ==="
echo "To activate: source ~/miniconda3/etc/profile.d/conda.sh && conda activate gdl"
echo "Project dir : $PROJECT_DIR"
