<p align="center">
  <img src="assets/logo.png" alt="CoughSense logo" width="460"/>
</p>

<h1 align="center">CoughSense</h1>
<p align="center"><b>AI-Based Cough Acoustic Analysis for Early Respiratory Disease Screening</b></p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue" alt="Python"/>
  <img src="https://img.shields.io/badge/ML-RandomForest%20%7C%20XGBoost-green" alt="ML"/>
  <img src="https://img.shields.io/badge/DL-CNN%20(PyTorch)-orange" alt="DL"/>
  <img src="https://img.shields.io/badge/Best%20Accuracy-85.1%25-brightgreen" alt="Accuracy"/>
  <img src="https://img.shields.io/badge/API-FastAPI-teal" alt="API"/>
  <img src="https://img.shields.io/badge/status-prototype-yellow" alt="Status"/>
</p>

<p align="center"><b>Team Auscultate</b> — Aryan Verma (B.Tech AI) &amp; Arfa Alam (B.Tech Civil Engineering)</p>

---

## Table of Contents

- [Overview](#overview)
- [The Problem](#the-problem)
- [Our Solution](#our-solution)
- [Key Value Points](#key-value-points)
- [How It Works](#how-it-works)
- [Results](#results)
- [Explainability](#explainability)
- [Data Quality Experiment](#data-quality-experiment)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Team Contributions](#team-contributions)
- [Setup &amp; Installation](#setup--installation)
- [Running the App](#running-the-app)
- [Testing &amp; Validation](#testing--validation)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Disclaimer](#disclaimer)

---

## Overview

**CoughSense** is a machine-learning and deep-learning system that screens for likely respiratory
conditions (such as COVID-19) from a short cough recording. It needs nothing more than a smartphone
microphone — no lab, no clinic visit, no cost. It is a **screening aid**, not a diagnosis: it answers
the narrow, high-value question *"should this person get tested?"*

The project deliberately implements **both classical ML and deep learning side by side** and compares
them honestly — a strong demonstration of how the two approaches behave on small, real clinical data.

---

## The Problem

Access to respiratory screening is limited in low-resource settings: clinics are far, tests cost money,
and results take time. Yet the cough itself carries information — clinicians have long used cough
character (wet vs. dry, productive vs. barking) as a diagnostic cue. If a machine can learn those same
acoustic patterns, screening becomes as accessible as a phone call.

---

## Our Solution

A pipeline that takes a cough clip and returns a prediction from three models, with confidence scores
and a clear medical disclaimer, exposed through both a REST API and a browser interface:

1. **Feature extraction** — MFCCs, spectral features, zero-crossing rate, RMS energy (for classical ML);
   log-mel spectrograms (for the CNN).
2. **Three models** — Random Forest, XGBoost, and a compact CNN — trained and cross-validated.
3. **Explainability** — SHAP analysis showing *why* the model predicts what it does.
4. **Web app** — a live screening tool + an analytics dashboard.

---

## Key Value Points

- **Real-world impact** — cheap, instant respiratory pre-screening for places with no easy lab access.
- **Underexplored modality** — audio biomarkers get far less attention than image or text ML, despite
  solid clinical grounding in cough acoustics.
- **Honest science** — every number is 5-fold cross-validated and reported with its standard deviation,
  never a single lucky split.
- **Explainable, not a black box** — SHAP shows which acoustic features drive each decision, and they
  match clinical intuition (cough timbre).
- **ML vs DL comparison** — demonstrates a real, well-documented tradeoff: on small clinical data,
  gradient-boosted trees beat deep learning.
- **A genuine data-quality experiment** — we tested scaling the data and learned *why quality beats
  quantity* in medical ML (details below).
- **Deployable** — runs fully locally on free tooling; FastAPI backend + static frontend.

---

## How It Works

```
 cough.mp3 ──► feature extraction ──►  ┌── Random Forest ─┐
                (MFCC / spectral)      │                  │
              ──► mel-spectrogram ──►  ├── XGBoost  ──────┤──► prediction + confidence
                                       │                  │      (+ SHAP explanation)
                                       └── CNN ───────────┘
```

1. Audio is resampled to 22.05 kHz and standardized to a fixed length.
2. Two representations are computed: a 37-value statistical feature vector (for the tree models) and a
   128×87 log-mel spectrogram (for the CNN).
3. Each model predicts COVID vs. healthy; the API returns all three with confidence scores.
4. Training data is expanded 4× with augmentation (noise, pitch-shift, time-shift) — applied only to the
   training split, after the train/test partition, to avoid leakage.

---

## Results

5-fold stratified cross-validation on the **Virufy clinical dataset** (121 clips: 48 COVID, 73 healthy):

| Model | Accuracy | Std Dev | Type |
|---|---|---|---|
| **XGBoost** | **85.1%** | ±6.2% | Classical ML (best) |
| Random Forest | 83.5% | ±6.9% | Classical ML |
| CNN (augmented) | 78.5% | ±9.3% | Deep Learning |

> Always cite the number with its spread — e.g. **"85.1% ± 6.2% (5-fold CV)"**. On a 121-clip dataset a
> single train/test split is too noisy to trust; cross-validation is the honest metric.

**Why the CNN trails the tree models:** with ~100 training clips, a CNN can't learn robust spectrogram
patterns the way it would with thousands of samples. Hand-crafted MFCC features inject decades of
audio-engineering knowledge, letting tree models generalize from far less data — a well-documented
tradeoff in applied ML, not a bug.

---

## Explainability

We apply **SHAP** (SHapley Additive exPlanations) to the XGBoost model to quantify each feature's
contribution. The analysis confirms that **MFCC-based timbral features carry the strongest signal** —
which aligns with clinical intuition: the "wet vs. dry" quality a doctor listens for is exactly what
these coefficients encode. The model is learning something real, not a spurious artifact.

Figures (in `reports/figures/`): SHAP summary &amp; bar plots, ROC curves (XGBoost AUC ≈ 0.93),
confusion matrices.

---

## Data Quality Experiment

We tested whether **more data** would help by scaling from 121 clinical clips to 321 using COUGHVID
(a 30,000-clip crowdsourced corpus). Counter-intuitively, **accuracy dropped from 85% to ~65%**.

Investigation showed why: COUGHVID labels are *self-reported* and inherently noisy, while Virufy labels
are *laboratory-confirmed*. Even after quality filters (`cough_detected ≥ 0.8`, SNR sorting) the recovery
was small. We also tried a Random-Forest + XGBoost ensemble and an enriched 102-feature set — neither beat
the simple 37-feature XGBoost.

> **Lesson: in medical ML, data quality beats quantity.** The final system uses the clean clinical data.

This experiment is itself a strength — it shows real experimentation, investigation, and a reasoned
engineering decision rather than blindly stacking data.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| Classical ML | scikit-learn (Random Forest), XGBoost |
| Deep Learning | PyTorch (CNN) |
| Audio | librosa, soundfile, ffmpeg |
| Explainability | SHAP |
| Backend | FastAPI + Uvicorn |
| Frontend | HTML / CSS / JavaScript (Web Audio API) |
| Visualization | matplotlib |

---

## Project Structure

```
cough-detect/
├── assets/                 # logo
├── data/
│   ├── raw/                # cough clips, one folder per class (covid/, healthy/)
│   └── processed/          # extracted features + spectrograms
├── models/                 # trained models (RF, XGBoost, CNN)
├── frontend/
│   ├── index.html          # live screening tool
│   ├── dashboard.html      # analytics dashboard
│   └── favicon.svg
├── reports/
│   ├── CoughSense_Technical_Report.docx
│   ├── figures/            # SHAP, ROC, confusion matrices
│   └── test_log.csv        # validation log
├── src/
│   ├── features.py             # feature extraction
│   ├── build_dataset.py        # build feature datasets
│   ├── ml_baseline.py          # train RF + XGBoost
│   ├── dl_model.py             # CNN architecture + training
│   ├── augment.py              # audio augmentation
│   ├── build_augmented_dataset.py
│   ├── dl_model_augmented.py   # train CNN on augmented data
│   ├── cross_validate.py       # 5-fold CV for all models
│   ├── ensemble.py             # RF + XGBoost ensemble experiment
│   ├── enhanced_features.py    # richer-feature experiment
│   ├── explain.py              # SHAP + ROC + confusion plots
│   ├── test_log.py             # testing & validation log
│   ├── serve.py                # FastAPI inference API
│   ├── download_more_data.py   # optional COUGHVID downloader
│   ├── reset_coughvid.py       # revert to clinical-only data
│   └── run_all.py              # runs the whole pipeline in order
├── requirements.txt
└── README.md
```

---

## Team Contributions

**Aryan Verma** — *Machine Learning &amp; Deep Learning lead.* Audio feature extraction (MFCCs, spectral
features), Random Forest / XGBoost / CNN model design and training, 5-fold cross-validation, data
augmentation, ensemble and enhanced-feature experiments, SHAP explainability, and the FastAPI inference
backend.

**Arfa Alam** — *Frontend, documentation &amp; validation.* Web interface (screening tool + analytics
dashboard UI/UX), project documentation (technical report and README), testing &amp; validation (running
cough samples through the system and logging predictions), and dataset organization (sorting and cleaning
cough clips into class folders).

---

## Setup & Installation

### Windows (recommended: Python 3.11)

```powershell
# 1. Virtual environment (3.11 — some libs don't build cleanly on 3.13/3.14 yet)
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
#    If activation is blocked: Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# 2. Dependencies
pip install -r requirements.txt

# 3. ffmpeg (needed to decode .mp3/.webm). If you don't have it and winget is unavailable:
pip install imageio-ffmpeg
python -c "import imageio_ffmpeg,shutil,os; exe=imageio_ffmpeg.get_ffmpeg_exe(); d=os.path.dirname(exe); shutil.copy(exe, os.path.join(d,'ffmpeg.exe')); print('ffmpeg ready at', d)"
#    then add that printed folder to PATH for the session:
#    $env:PATH = "<that folder>;" + $env:PATH
```

### Any OS (quick)

```bash
pip install -r requirements.txt      # use a venv
cd src
python run_all.py                    # runs the whole pipeline in order
```

`run_all.py` chains every stage (features → train RF/XGBoost → augment → train CNN → cross-validate →
SHAP/plots). Run stages individually any time — see `src/`.

---

## Running the App

The web interface needs **two terminals**.

**Terminal 1 — backend (API):**
```powershell
cd src
uvicorn serve:app --reload --port 8000
# wait for "Application startup complete"
```

**Terminal 2 — frontend:**
```powershell
cd frontend
python -m http.server 5500
```

Open **http://localhost:5500** in your browser. Look for the **API ONLINE** badge (top-right). Upload or
record a cough to get a live prediction; click **Dashboard** for the analytics view.

> Windows note: activate the venv in each new terminal and re-apply the ffmpeg PATH line if audio
> decoding fails.

---

## Testing & Validation

Run a batch of clips through all three models and log the predictions:

```bash
cd src
python test_log.py --n 20
```

Prints a per-clip table (actual vs. each model's prediction vs. consensus) and saves
`reports/test_log.csv`. On a 20-clip sample, RF and XGBoost each scored 90% and the 3-model consensus
90% — consistent with the cross-validated results.

---

## Limitations

- **Small, single-source dataset** — not yet validated across populations, devices, or other respiratory
  conditions. Results are a proof of concept for the *approach*, not a clinical-grade claim.
- **Binary scope** — currently COVID vs. healthy; more conditions need more labeled audio.
- **Screening, not diagnosis** — the tool detects acoustic patterns, not disease. The medical disclaimer
  must stay in any demo.

---

## Future Work

- Larger, multi-source clinical datasets to close the CNN gap.
- Multimodal fusion — combine cough audio with a short self-reported symptom form.
- Grad-CAM visualizations of the spectrogram regions the CNN attends to.
- Extend to additional respiratory conditions (asthma, bronchitis, TB).
- A guidance layer that turns a screening result into clear next steps.

---

## Disclaimer

CoughSense is a **research prototype and screening aid**, not a medical device and not a diagnosis.
It must not be used to make health decisions. Always consult a qualified healthcare professional.

---

<p align="center"><i>Team Auscultate — CoughSense — Not for clinical use</i></p>