# Gaussian Processes for Modelling Double Gene Knockout


### Setup

**Option 1: Conda**
```
conda create -n gpgenes python=3.13
conda activate gpgenes
pip install -r requirements.txt
```

**Option 2: venv**
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Minimal GRN simulator + GP residual model

Run:
python -m gpgenes.experiments.train_gp


This simulates control/single/double gene knockouts with replicates, computes control baseline, and fits per-gene GP models to residuals using a graph-informed multi-hot kernel.
