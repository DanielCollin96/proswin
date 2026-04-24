# PROSWIN: Probabilistic Solar Wind Speed Forecasting

Official repository for [PROSWIN: Probabilistic Solar Wind Speed Forecasting Using Deep Distributional Regression From Solar Images](https://doi.org/TBD), published in *TBD* (2026) by [Daniel Collin](https://www.linkedin.com/in/daniel-collin-52abb0205/), [Yuri Shprits](https://www.gfz.de/staff/yuri.shprits), [Luca Chiarabini](https://www.linkedin.com/in/luca-chiarabini-596a07195/) [Stefan J. Hofmeister](https://www.astro.columbia.edu/content/stefan-hofmeister), [Nadja Klein](https://www.kleinlab.de/subpages_team/klein.html) and [Guillermo Gallego](http://www.guillermogallego.es).

<div align="center">
  <img src="assets/eyecatcher.png" alt="Alt Text">
</div>



<h2 align="center">

[Journal Publication](https://doi.org/TBD) | [Data Publication](https://doi.org/10.5880/GFZ.OJSJ.2026.001) | [Solar Image Repo](https://github.com/DanielCollin96/solar_image_processing)

</h2>

---
## Overview

PROSWIN is a probabilistic deep-learning model for forecasting the solar wind speed at L1. It combines SDO/AIA solar images with physical features about solar wind conditions from the previous solar rotation and the state of the solar cycle to predict conditional log-normal distributions of the solar wind speed with a lead time of 4 days. This repository provides code for model training and evaluation of predictions. With the provided [dataset](https://doi.org/10.5880/GFZ.OJSJ.2026.001), all results from the paper can be reproduced.

---

## Model Architecture

<div align="center">
  <img src="assets/model_pipeline.png" alt="Alt Text">
</div>

PROSWIN consists of an image encoding backbone and two prediction heads, predicting the two distributional parameters mu and sigma of a lognormal distribution.
The model is trained in two sequential stages.


### Backbone
A pretrained Swin Transformer processes stacked solar images and magnetograms. The encoded feature vector is concatenated with the physical feature vector.

### Prediction Heads

Each prediction head of made of two fully-connected neural network layers with dropout. One head predicts the mu parameter and the other one the sigma parameter of the predictive lognormal distribution.

### Two-stage training
| Stage     | Trained part      | Loss                    | Output                                             |
|-----------|-------------------|-------------------------|----------------------------------------------------|
| **mu**    | Image encoder + mu head | Weighted MSE            | Expected solar wind speed value (conditional mean) |
| **sigma** | sigma head | Negative log-likelihood | Conditional lognormal distribution                 |

The image encoder and mu head are frozen during sigma training, 
so the uncertainty estimate is built on top of a fixed point forecast. 
This is possible due to a data transformation, which maps the targets to a normal distribution (log transform), 
such that mean (mu) and variance (sigma) decouple. 
The weighted MSE emphasises high-speed stream events.

### Predictive distribution
Network outputs are in a standardised log-space. 
Inverting the scaler and re-parameterising yields a shifted log-normal distribution with parameters (shape, scale, loc).

### Physical features

- Solar wind speed from the previous solar rotation (hourly values from 28 to 26 days ago)
- Heliospheric latitude of the Earth relative to the equatorial plane of the Sun
- Monthly sunspot numbers over the last year
- Progress of the solar cycle in years

---

## Getting Started

This project uses [uv](https://docs.astral.sh/uv/) for extremely fast Python package and project management.

### 0. Hardware Prerequisites

- For training: CUDA-capable GPU (strongly recommended; CPU-only runs are supported but slow)
- For inference or evaluation: CPU is sufficient

### 1. Install uv

If you don't have uv installed yet, install it:

#### Linux / MacOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Windows (Powershell)

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Verify the installation:

```bash
uv --version
```

### 2. Project Setup

After cloning the repository, navigate to the project directory and sync the dependencies. 
This will automatically create a virtual environment (`.venv`) and install all required packages from the `uv.lock` file.

```bash
# Clone the repository
git clone https://github.com/DanielCollin96/proswin.git
cd proswin

# Create a virtual environment and install all dependencies
uv sync
```
The package is installed in editable mode from `src/`, so `import proswin` resolves correctly from any working directory.


### 3. Running the Project

You do not need to activate the virtual environment manually. 
Use `uv run` to execute scripts within the project context:

```bash
uv run <script>.py
```

### 4. Optional: Weights & Biases experiment tracking

Log into your Weights & Biases account:

```bash
wandb login
```

Then set `wandb_project: your_project_name` in `configs/config.yaml`. Leave it blank or omit to disable W&B entirely.

---

## Data Preparation

The dataset used in the paper is available from GFZ Data Services here: https://doi.org/10.5880/GFZ.OJSJ.2026.001

### Expected directory layout

After unpacking, update the `paths` section of `configs/config.yaml` to point to the following directories:

```
<data_root>/
├── solar_images/              # SDO/AIA EUV images (.npy, one file per hour)
│   ├── 171/                   # AIA 171 Å channel
│   │   └── <year>/<month>/
│   ├── 193/                   # AIA 193 Å channel
│   ├── 211/                   # AIA 211 Å channel
│   └── HMI/                   # HMI line-of-sight magnetograms
├── machine_learning/        
│   ├── ml_dataset.csv         # Feature values
│   └── literature_models.csv  # Needed for literature comparison
├── physical_features/         # Scalar time series (CSV)
│   ├── solar_wind_speed.csv   # Needed for baseline model comparison
│   ├── sunspot_number.csv     # Needed for solar cycle plot
│   └── ch_area.csv            # Needed for baseline model comparison
└── solar_events/              # Event catalogues (CSV)
    ├── cme_list.csv
    ├── hss_list.csv
    └── enhancement_list.csv

```

To reuse/evaluate trained models or predictions, copy models and predictions into the `results` directory:

```
<results_root>/
├── models/                     # Trained models
│   ├── full_model/             # Models trained with all available features for each CV fold      
│   │   └── <model>_<fold>.pth
│   ├── image_only/             # Models trained without physical features for each CV fold
│   │   └── <model>_<fold>.pth
└── predictions/                # Computed predictions
    ├── <model>_full_model.csv  # Predictions of full models agggregated over all CV folds   
    └── <model>_image_only.csv  # Predictions of image-only models agggregated over all CV folds  


```

---
## Running the Code

All scripts read exclusively from `configs/config.yaml`. No command-line arguments are used. 
Edit the configuration file before each run to set up the desired model:
- Set all `paths` entries to your data directories
- Configure the dataset and input data (e.g., image channels) under `dataset`
- Set model hyperparameters and the model's name to identify this run's outputs under `model`
- Set training hyperparameters for the mu and sigma stages under `training`
- Set `use_multiple_gpus: true` to run the training with DDP on multiple GPUs
- Optionally set `wandb_project` for experiment tracking

### Model Training

To run a simple training iteration on a fixed train/val/test split, 
edit `scripts/train_single_split.py` and set `TRAIN_DATES`, `VAL_DATES`, and `TEST_DATES`
with your the desired training, validation and test dates, respectively. Then run:

```bash
uv run scripts/train_single_split.py
```

### Prediction

To use a trained model for inference, edit `scripts/predict.py` and set `MODEL_PATH` to the path to the saved model checkpoint,
and set the time range for prediction at `PREDICTION_START` and `PREDICTION_END`. Then run:

```bash
uv run scripts/predict.py
```

### Evaluation

To evaluate saved predictions and produce evaluation plots, edit `scripts/evaluate_predictions.py` and set `PREDICTION_PATH`
to the path to the saved probabilistic predictions. Then, run:

```bash
uv run scripts/evaluate_predictions.py
```


### How to use Probabilistic Predictions

The probabilistic predictions are based on [`scipy.stats.lognorm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html)
under the hood.
You can extract the `lognorm` parameters using `prediction.shape`, `prediction.scale`, and `prediction.loc`,
or extract quantities of interest, for example:

```python
prediction.get_expected_values()             # mean
prediction.get_quantiles(0.5)                # quantiles, e.g., median
prediction.get_quantiles([0.025, 0.975])     # 95% confidence interval
prediction.get_cdfs(500)                     # value of cumulative distribution function at 500 km/s
prediction.get_pdfs(500)                     # value of probability density function at 500 km/s
prediction.get_crps(observation)             # per-sample CRPS 
```
For full API details see [`predictions.py`](https://github.com/DanielCollin96/proswin/blob/main/src/proswin/predictions.py).


### Reproducing the Experiments from the Paper


#### Cross-Validation

To perform the full cross-validation, run:

```bash
uv run scripts/cross_validation.py
```

This pipeline executes the following:
1. Builds and caches the dataset
2. Generates the 5-fold cross-validation split
3. Trains each fold in two stages (mu then sigma)
4. Evaluates each fold and aggregates predictions across all test periods
5. Saves all computed results into the `results/` subdirectories.

#### Model Comparison

To compare all model variants tested in the paper and literature models, run:

```bash
uv run scripts/compare_models.py
```

---

## Citation
If you use this work in your research, please cite it as follows:

```bibtex
@article{Collin2026,
  title   = {PROSWIN: Probabilistic Solar Wind Speed Forecasting Using Deep Distributional Regression From Solar Images},
  author = {Collin, Daniel and Shprits, Yuri and  Chiarabini, Luca and Hofmeister, Stefan J. and Klein, Nadja and Gallego, Guillermo},
  journal = {TBD},
  year    = {2026},
  url     = {https://doi.org/TBD}
}
```

If you use the dataset, please cite:

```bibtex
@misc{Collin2026dataset,
  title = {Solar Images and Physical Features for Probabilistic Solar Wind Speed Prediction},
  author = {Collin, Daniel and Shprits, Yuri and  Chiarabini, Luca and Hofmeister, Stefan J. and Klein, Nadja and Gallego, Guillermo},
  publisher = {GFZ Data Services},
  year = {2026},
  howpublished = {Data Set},
  doi = {10.5880/GFZ.OJSJ.2026.001}
}
```
---


## Code developer

[Daniel Collin](https://www.linkedin.com/in/daniel-collin-52abb0205/). 

In case of questions or bugs, please contact Daniel at collin@gfz.de.

---


## License

This work is released under [MIT License](https://github.com/DanielCollin96/proswin/blob/main/LICENSE).
