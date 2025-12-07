# Fraud Detection with PyOD AutoEncoder

This project implements an **unsupervised deep learning approach** to detect credit card fraud using an AutoEncoder from the PyOD library.

## Dataset

The Kaggle dataset is downloaded **automatically** using `kagglehub` at runtime:

~/.cache/kagglehub/datasets/mlg-ulb/creditcardfraud/

No manual setup or CSV download is required.

## How to Run

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run script
python3 fraud_detection_autoencoder.py
The script will:

Download the dataset (if not cached)

Train the AutoEncoder

Print evaluation metrics

Display plots
```
Output

Training logs

Classification report

Confusion matrix

ROC curve

Files
```
fraud_detection_autoencoder.py
requirements.txt
README.md
