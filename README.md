# Lab 1: Supervised Learning - EEG Mental State Classification

## Project Overview

This project implements binary emotion classification from EEG signals using supervised machine learning algorithms. The goal is to distinguish between **focused** and **unfocused/drowsed** mental states using traditional ML approaches.

## Problem Statement

The task involves classifying EEG signals into two states:
- **Class 0**: Focused mental state
- **Class 1**: Unfocused/Drowsed mental state

This problem has real-world applications in driver drowsiness detection, attention assessment, human-computer interaction, and healthcare monitoring.

## Dataset

**SEED-IV Dataset** from Shanghai Jiao Tong University
- **Source**: https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html
- **Subjects**: 15 participants
- **Sessions**: 3 sessions per subject (recorded on different days)
- **Trials**: 24 trials per subject
- **Channels**: 62-channel ESI NeuroScan System
- **Sampling Rate**: 128 Hz
- **Total Samples**: 25,794 samples
- **Features**: 2,232 features (62 channels × 36 frequency bands)

### Data Preprocessing
- Feature extraction using Short-Time Fourier Transform (STFT)
- Frequency bands: 4-40 Hz (36 bands)
- StandardScaler normalization applied

## Methodology

### Models Implemented
1. **Logistic Regression** - Linear classification model
2. **Support Vector Machine (SVM)** - RBF kernel for non-linear classification
3. **Random Forest** - Ensemble method with 200 estimators

### Evaluation Strategy
- **Cross-Validation**: 3-fold leave-one-session-out
- **Class Balancing**: Balanced class weights using `compute_class_weight`
- **Hyperparameter Tuning**: GridSearchCV for all models

### Key Features
- Comprehensive EDA (Exploratory Data Analysis)
- Multiple model comparison
- Hyperparameter optimization
- Detailed performance metrics

## Results

| Model | Mean Accuracy | Best Performer |
|-------|--------------|----------------|
| **SVM (RBF)** | **58.37%** | ✓ |
| Random Forest | 55.86% | |
| Logistic Regression | 50.07% | |

**Key Findings:**
- SVM with RBF kernel performed best, demonstrating the importance of non-linear models for EEG classification
- Logistic Regression achieved essentially random performance (50%), highlighting the need for non-linear approaches
- Results are reasonable given the complexity of EEG-based mental state classification

## Installation & Setup

### Prerequisites
- Python 3.8+
- Jupyter Notebook

### Dependencies
Install required packages:
```bash
pip install -r requirements.txt
```

### Dataset Setup
1. Download SEED-IV dataset from: https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html
2. Extract the dataset to a `SEED_IV/` folder in the project directory
3. Ensure the folder structure: `SEED_IV/eeg_raw_data/{session}/{subject}.mat`

### Running the Notebook
1. Open `lab_1.ipynb` in Jupyter Notebook
2. Run all cells sequentially
3. The notebook includes:
   - Data loading and preprocessing
   - Exploratory Data Analysis (EDA)
   - Model training and evaluation
   - Results visualization
   - Discussion and analysis

## Project Structure

```
Lab_1/
├── README.md
├── lab_1.ipynb          # Main notebook
├── loading_data.py      # Data loading utilities
├── CONSTANT.py          # Configuration constants
└── requirements.txt     # Python dependencies
```

## Key Concepts

- **Short-Time Fourier Transform (STFT)**: Frequency-domain feature extraction
- **Leave-One-Session-Out Cross-Validation**: Prevents data leakage across sessions
- **Class Imbalance Handling**: Balanced class weights for fair evaluation
- **Hyperparameter Tuning**: GridSearchCV for model optimization

## References

- **Dataset**: SEED-IV Dataset - https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html
- **Key Concepts**: CU Boulder's Supervised Learning Course Note

## Author

Completed as part of the Supervised Learning Final Project.

