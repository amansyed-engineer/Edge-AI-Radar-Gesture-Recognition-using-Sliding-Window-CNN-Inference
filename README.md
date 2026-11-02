# Radar-Based Kick Gesture Recognition Using Temporal Segmentation and Deep Learning

## Project Overview
This project investigates a radar-based approach for reliable kick gesture recognition, aimed at hands-free automotive access systems. The work focuses on accurately identifying the temporal region of a kick gesture within long radar recordings and evaluating the impact of different temporal segmentation strategies on classification performance.

An FMCW radar sensor is used to capture complex baseband I/Q signals, which are processed using signal processing techniques and classified using a deep learning model. The system is evaluated under a person-independent setup to ensure realistic generalization to unseen users.



## Objectives
- Detect and localize kick gestures in long radar time-series data  
- Compare multiple temporal segmentation techniques  
- Train a deep learning model for binary kick gesture classification  
- Evaluate system performance using person-wise subject splits  



## Methodology Overview
The proposed framework follows an end-to-end processing pipeline:

1. Radar data acquisition using FMCW radar  
2. Signal preprocessing (bandpass filtering and normalization)  
3. Temporal segmentation to locate the kick gesture  
4. Fixed-length window extraction  
5. CNN-based gesture classification  
6. Person-wise training and sliding-window evaluation  



## Temporal Segmentation Techniques
Three segmentation methods are implemented and compared:

- **Energy-Based Segmentation**  
  Identifies the kick gesture by detecting peaks in short-time signal energy.

- **Gradient-Based Segmentation**  
  Locates the onset of the kick by identifying rapid temporal changes in signal magnitude.

- **Doppler-Based (STFT) Segmentation**  
  Uses time–frequency analysis to capture motion-induced Doppler shifts and identify peak velocity regions.



## Classification Model
A one-dimensional Convolutional Neural Network (1D CNN) is used for binary classification of segmented radar windows into:
- **Valid Kick**
- **Invalid Motion**



## Evaluation Strategy
- Person-wise 80/20 train–test split  
- Test data consists exclusively of unseen subjects  
- Sliding-window inference during testing  
- Performance metrics:
  - Accuracy (ACC)
  - Precision
  - Recall
  - False Positive Rate (FPR)
  - ROC–AUC  



## Key Results
- Temporal segmentation significantly affects classification performance  
- Doppler-based segmentation achieves the most robust and consistent results  
- Gradient-based segmentation provides a good balance between accuracy and complexity  
- A window length of **2000 samples** offers the best overall performance  


## Repository Structure

```
Radar-Based-Kick-Gesture-Recognition/
│
├── Code/
│   ├── Project_Thesis_Radar_gesture_Data_Segmentation.ipynb
│   └── README_code.md
│
├── Report/
│   ├── Project_Thesis.pdf
│   └── Project_Summary.pdf
│
├── Presentation/
│   └── Project_thesis.pptx
│
├── Results/
│   ├── figures/
│   │   ├── segmentation_examples/
│   │   ├── roc_curves/
│   │   ├── confusion_matrices/
│   │   └── training_curves/
│   │
│   ├── performance_tables/
│   │   └── method_comparison.pdf
│   │
│   └── README_results.md
│
├── README.md
└── requirements.txt
```


## How to Run
1. Install dependencies listed in `requirements.txt`  
2. Open the Jupyter notebook in the `Code/` directory  
3. Execute the notebook sequentially to reproduce preprocessing, segmentation, training, and evaluation steps  


## Deliverables
- Complete source code implementation  
- Project thesis document  
- Project summary report  
- Final presentation slides  
- Experimental results and analysis  


## Conclusion
This project demonstrates that combining physically meaningful temporal segmentation with deep learning enables robust and person-independent radar-based kick gesture recognition.

