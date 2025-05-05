# 📦 Machine Learning App on Mendel OS (Edge T Device)

This project runs a Python-based machine learning application directly on a Mendel OS edge device (e.g., Coral Dev Board with NXP i.MX 8M ARM processor). The app performs real-time predictions using pre-trained models.

---

## ⚙️ Environment Setup

Ensure your Mendel OS device has internet access. Then, follow these steps to install the required Python packages system-wide.

### 🛠 Step 1: Run Installation Script

Execute the setup script to install Python 3 and the required packages:

```bash
chmod +x install_packages.sh
./install.packages.sh

cd src
mkdir results
python3 main.py
