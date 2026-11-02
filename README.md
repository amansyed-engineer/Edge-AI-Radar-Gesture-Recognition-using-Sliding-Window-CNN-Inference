# Radar-Based Kick Gesture Recognition

Project thesis at OTH Regensburg — Aman Syed

This project studies how temporal segmentation affects radar-based kick gesture recognition for hands-free vehicle access. It compares energy, gradient and Doppler (STFT) segmentation, followed by a 1D convolutional neural network that classifies valid kicks and invalid movements.

The repository brings together the experiment notebook, its Python script version, the thesis, presentation and recorded results.

## Approach

The pipeline loads four radar I/Q channels, applies bandpass filtering and amplitude normalisation, and extracts fixed-length segments around detected motion. The experiments compare window lengths of 1000, 2000 and 4000 samples. Subjects are separated for training, validation and testing.

The thesis reports that Doppler-based segmentation gives the most consistent performance, with a 2000-sample window providing the best overall balance. The detailed comparisons are in the [thesis](Report/Project_Thesis.pdf) and [results tables](Results/Performance-Tables/Method_Comparison_Tables.pdf).

## Files

| Folder | Contents |
| --- | --- |
| [Code](Code/readme.md) | Jupyter notebook and matching Python script |
| [Report](Report/Project_Thesis.pdf) | Full thesis and project summary |
| [Presentation](Presentation/Project_Thesis_Presentation.pdf) | Presentation PDF and editable PowerPoint slides |
| [Results](Results/readme.md) | Segmentation figures, training curves, ROC curves, confusion matrices and comparison tables |

```text
Code/
  Project_Thesis_Radar_gesture_Data_Segmentation.ipynb
  Project_Thesis.py
  readme.md
Report/
  Project_Thesis.pdf
  Project_Summary.pdf
Presentation/
  Project_Thesis_Presentation.pdf
  Project_thesis.pptx
Results/
  Figures/
    Confusion_Matrices.pdf
    ROC_Curves.pdf
    Segmentation_Results.pdf
    Training_Curves.pdf
  Performance-Tables/
    Method_Comparison_Tables.pdf
  readme.md
README.md
requirements.txt
```

## Running the experiments

Create and activate a Python environment, then install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

The radar dataset is not included. Set `RADAR_DATA_DIR` to its location before starting Python or Jupyter. On macOS or Linux:

```bash
export RADAR_DATA_DIR="/absolute/path/to/dataset"
python -m notebook
```

In PowerShell, use `$env:RADAR_DATA_DIR = "C:/path/to/dataset"` before starting Jupyter. You can also set `input_dir` directly in the notebook's data-loading cell or in the script. Without a setting, the code looks for `data60` in the current working directory.

Open [the notebook](Code/Project_Thesis_Radar_gesture_Data_Segmentation.ipynb) and run the cells in order. To run the script instead, use:

```bash
python Code/Project_Thesis.py
```

The loader expects one folder per person, with comma-separated recordings containing four I/Q channels. Recording names must encode the gesture and integer repetition as `gesture__repetition`, as in the original dataset. Person folders whose names start with a digit are skipped by the current loader. See the [code notes](Code/readme.md) before using another dataset.

## Reproducing the reported results

The notebook outputs and PDF figures are records of the original experiments. Installing the dependencies alone will not reproduce them: the original dataset, subject selection and environment are also needed. Dependency versions here are minimum requirements; the original environment was not locked.

The supplied implementation has a few limitations to account for:

- The thesis describes an 80/20 person-wise split, but the supplied code loads the first 10 people and assigns IDs 2, 5 and 8 to testing. With all 10 people loaded, that is a 70/30 split before validation and any discarded recordings. IDs depend on directory iteration order, so record the person-to-ID mapping when running it.
- The test loop passes already segmented inputs of length `L` to the sliding-window helper. This produces one prediction per segment; it does not scan the full 12000-sample recording in that loop.
- Filtering and normalisation happen before the subject split. Filtering is applied across the concatenated recordings, and normalisation uses the full dataset. A stricter evaluation would filter recordings separately and fit normalisation using training subjects only.

Changes to those evaluation steps need a fresh experiment run and new result tables. The archived reports and figures have been kept unchanged.

## Documents

- [Full thesis](Report/Project_Thesis.pdf)
- [Project summary](Report/Project_Summary.pdf)
- [Presentation PDF](Presentation/Project_Thesis_Presentation.pdf)
- [Editable presentation](Presentation/Project_thesis.pptx)

## Author

Aman Syed · OTH Regensburg
