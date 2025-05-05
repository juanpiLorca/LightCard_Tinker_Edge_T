#!/bin/bash

echo ">>> Updating package lists..."
sudo apt-get update

echo ">>> Installing Python 3, pip, and scientific libraries..."
sudo apt-get install -y \
    python3-pip \
    python3-numpy \
    python3-scipy \
    python3-sklearn \
    python3-sklearn-lib \
    python3-joblib \
    python3-pandas \
    build-essential \
    libpython3-dev

echo ">>> Verifying installations..."
python3 -c "import numpy, sklearn, scipy, joblib, pandas; print('✅ All packages installed successfully.')"
