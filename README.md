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
.
├── Code/
│   ├── Project_Thesis_Radar_gesture_Data_Segmentation.ipynb
│   ├── Project_Thesis.py
│   └── README.md
├── Report/
│   ├── Project_Thesis.pdf
│   └── Project_Summary.pdf
├── Presentation/
│   ├── Project_thesis.pptx
│   └── Project_Thesis_Presentation.pdf
├── Results/
│   ├── Figures/
│   │   ├── Confusion_Matrices.pdf
│   │   ├── ROC_Curves.pdf
│   │   ├── Segmentation_Results.pdf
│   │   └── Training_Curves.pdf
│   ├── Performance-Tables/
│   │   └── Method_Comparison_Tables.pdf
│   └── README.md
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```


## How to Run
1. Create a Python environment and install dependencies with `python -m pip install -r requirements.txt`. The original environment versions were not recorded; these are minimum requirements, not a locked environment.
2. Obtain the radar dataset separately; it is not included in this repository.
3. Set `input_dir` in the data-loading cell to your dataset directory. It must contain one folder per person, with CSV recordings containing four I/Q columns. The loader parses filenames as `gesture__repetition` and skips person folders starting with a digit.
4. Open [the notebook](Code/Project_Thesis_Radar_gesture_Data_Segmentation.ipynb) and execute its cells in order. [Project_Thesis.py](Code/Project_Thesis.py) contains the same experiment code as a standalone script; set its `input_dir` before running it.
5. Review the subject selection before reproducing results: the supplied code loads only 10 people (`pers=10`) but uses a test-ID list intended for a larger dataset. Of the 10 loaded IDs (0–9), 2, 5 and 8 are test subjects, giving a 70/30 split. Person IDs depend on directory iteration order. The 80/20 description above is the reported study protocol, not the default supplied-code split.

The code and recorded outputs are preserved from the supplied thesis files. Training has not been rerun as part of repository organisation.  


## Deliverables
- Complete source code implementation  
- Project thesis document  
- Project summary report  
- Final presentation slides  
- Experimental results and analysis  


## Conclusion
This project demonstrates that combining physically meaningful temporal segmentation with deep learning enables robust and person-independent radar-based kick gesture recognition.



## Project Documents and Results
- [Full thesis](Report/Project_Thesis.pdf) and [project summary](Report/Project_Summary.pdf)
- [Presentation PDF](Presentation/Project_Thesis_Presentation.pdf) and [editable slides](Presentation/Project_thesis.pptx)
- [Results guide](Results/README.md) and [method comparison tables](Results/Performance-Tables/Method_Comparison_Tables.pdf)
- [Code guide](Code/README.md)
