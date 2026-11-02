# Experiment code

The [notebook](Project_Thesis_Radar_gesture_Data_Segmentation.ipynb) contains the experiment workflow and recorded outputs. [Project_Thesis.py](Project_Thesis.py) contains the same executable code as a script. Use either version; there is no need to run both.

The workflow covers data loading, I/Q preprocessing, energy, gradient and Doppler segmentation, CNN training, and person-wise evaluation. Shared function names are redefined for each segmentation method, so run the notebook cells in order.

## Dataset setup

Install the root [requirements](../requirements.txt) and set `RADAR_DATA_DIR` to the dataset location. Alternatively, edit `input_dir` in the data-loading cell or script. Setup commands are in the [main README](../README.md#running-the-experiments).

The current loader reads comma-separated files with a header row and four columns, which it assigns as `I_data_lower`, `Q_data_lower`, `I_data_upper` and `Q_data_upper`, in that order.

It expects one folder per person and recording names in the original dataset's `gesture__repetition` form, where repetition is an integer. Despite the legacy function's `mode` parameter, its present implementation reads CSV-formatted data; a separate 24 GHz TXT-loading path is not implemented.

## Before training

Review `pers=10`, `TEST_PERSONS`, the person-to-ID mapping, and `WINDOWS` before starting a run. The script displays plots and trains several models, so allow enough time and memory.

The [reproduction notes](../README.md#reproducing-the-reported-results) explain the split, preprocessing and evaluation limitations. Existing notebook outputs are retained from the original experiments and have not been regenerated for the setup changes.
