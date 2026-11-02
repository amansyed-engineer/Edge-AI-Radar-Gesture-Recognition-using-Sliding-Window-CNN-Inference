# Experimental Results and Analysis

This folder contains the experimental outputs generated during the evaluation of the radar-based kick gesture recognition system.

## Contents

### 1. Segmentation Examples
- Visualizations showing the temporal localization of kick gestures using:
  - Energy-based segmentation
  - Gradient-based segmentation
  - Doppler (STFT)-based segmentation
- These figures illustrate how each method identifies the kick region within long radar recordings.

### 2. Performance Evaluation
- Quantitative comparison of segmentation methods and window lengths.
- Metrics used for evaluation:
  - Accuracy (ACC)
  - Precision
  - Recall
  - False Positive Rate (FPR)
  - ROC–AUC

### 3. ROC Curves and Confusion Matrices
- ROC curves demonstrate the classification performance across decision thresholds.
- Confusion matrices show true positives, false positives, true negatives, and false negatives for unseen subjects.

### 4. Key Findings
- Doppler-based segmentation provides the most robust and consistent performance.
- Gradient-based segmentation offers a good trade-off between accuracy and computational complexity.
- A window length of 2000 samples achieves the best balance between information content and noise.

## Notes
- All results are obtained using a person-wise 80/20 train–test split.
- Test data consists exclusively of unseen subjects to ensure realistic generalization performance.
