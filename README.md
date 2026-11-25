# Dysarthric Speech Classification with Machine Learning

Dysarthric is a Python research project for binary classification of speech recordings in the context of dysarthria. It compares standard acoustic features with two additional descriptors, peak density and a measure of multifractal scaling, using the same neural network architecture across feature sets.

## Quick start

Use Python 3.10 or newer. From the repository root, create a virtual environment and install the package with its development dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m dysarthric_speech demo --output-dir runs/demo
```

The demo uses generated audio to exercise feature preparation, training, and evaluation. It requires no corpus access. Its metrics describe the synthetic example.

## Methodology

The pipeline extracts one feature vector per recording. Standard acoustic features are averaged over frames. Peak density and the multifractal descriptor use the waveform at its original sample rate.

| Variant | Features | Dimensions |
|---|---|---:|
| `normal` | 13 MFCCs, 12 chroma coefficients, 7 spectral contrast coefficients, and zero-crossing rate | 33 |
| `max_density` | The baseline plus the number of local maxima with amplitude at least zero divided by the number of waveform samples | 34 |
| `multifrac` | The peak-density variant plus Δα, calculated as α(−5) minus α(5) | 35 |

The multifractal calculation applies a causal moving average to the waveform, computes RMS fluctuations at scales of 32 to 4,096 samples, and estimates scaling exponents by log–log regression for moment orders −5 through 5. It uses the raw waveform directly. The original MF-DMA method starts with a cumulative profile. [Gu and Zhou, *Detrending moving average algorithm for multifractals*](https://arxiv.org/abs/1005.0877)

The classifier is a PyTorch multilayer perceptron with two 64-unit ReLU hidden layers and a sigmoid output. Training runs on CPU and uses full-batch Adam with a learning rate of 0.001 and binary cross-entropy. The default is 5,000 epochs. The protocol uses an 80/20 split by recording with split seed 42, applies SMOTE to the training set, and uses a classification threshold of 0.5.

## Run an experiment with local audio

Prepare a CSV with a `path` column containing recording paths relative to your data root and a `label` column containing binary labels. Keep a record of how your manifest maps `0` and `1` to groups.

```csv
path,label
recordings/example_a.wav,0
recordings/example_b.wav,1
```

This example shows the CSV schema. Training requires enough recordings of each class to form a split and apply SMOTE.

```bash
python -m dysarthric_speech prepare \
  --manifest dataset/manifest.csv --data-root dataset \
  --variant normal --output-dir runs/normal/data
python -m dysarthric_speech train \
  --input-dir runs/normal/data --variant normal --output-dir runs/normal/model --seed 42
python -m dysarthric_speech evaluate \
  --input-dir runs/normal/data --checkpoint runs/normal/model/model.pt \
  --output-dir runs/normal/evaluation
```

Preparation writes `dataset.npz` with the feature arrays, labels, and relative recording paths for each split. Training saves `model.pt`, the training loss plot `loss.png`, and the settings and dependency versions used for training in `run.json`. Training and evaluation both write `metrics.json`, `confusion.png`, and `confusion_percent.png`. Use one variant consistently for preparation and training. Keep the manifest and experiment outputs together so the data split can be traced back to the input, and use `--seed` to set the PyTorch training seed.

The `multifrac` variant can produce non-finite values for short or silent recordings. The pipeline reports invalid values so that a change to the dataset or the numerical method can be made explicitly.

### Dataset

The historical scripts reference an organized copy of TORGO. The corpus recordings and source manifest are external inputs. Check the terms of the edition you obtained before using or redistributing the recordings or derived artifacts. Cite Rudzicz, Namasivayam, and Wolff, *The TORGO database of acoustic and articulatory speech from speakers with dysarthria*, Language Resources and Evaluation 46, 523–541 (2012). [University of Toronto TORGO database](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html) · [Linguistic Data Consortium edition](https://catalog.ldc.upenn.edu/products/LDC2012S02)

## Historical results

The repository preserves the historical experiment scripts and artifacts under `train/`. The table reports metrics recomputed from the stored confusion matrices, each containing 3,527 test recordings. The majority-class reference is 65.41% accuracy.

| Historical run | Features | Accuracy | Balanced accuracy | Class 1 F1 |
|---|---:|---:|---:|---:|
| `train/normal/` | 33 | 95.04% | 94.43% | 92.80% |
| `train/max_density/` | 34 | 95.44% | 95.14% | 93.45% |
| `train/` | 34 | 95.04% | 94.86% | 92.93% |

The two 34-feature runs use the same saved tensors and different trained weights. The historical split is by recording and the stored tensors lack speaker and session IDs, so speaker-independent generalization remains unmeasured. The historical label-to-group mapping is undocumented. The multifractal tensors contain 55 training and 12 test rows with non-finite Δα, and there is no verified result for that variant. [Technical audit in Portuguese](docs/auditoria-tecnica.md)

## Project layout and development

| Path | Contents |
|---|---|
| `src/dysarthric_speech/` | Feature extraction, dataset preparation, model training, and evaluation |
| `tests/` | Behavior and integration checks |
| `train/` | Historical scripts, tensors, checkpoints, and plots |
| `docs/auditoria-tecnica.md` | Technical audit of the original repository |

```bash
python -m pytest
python -m ruff check src tests
```

The synthetic workflow was verified on Python 3.12 with CPU execution and the following direct dependency versions. The historical experiment environment was not recorded.

| Dependency | Verified version |
|---|---|
| NumPy | 2.0.0 |
| SciPy | 1.16.3 |
| librosa | 0.11.0 |
| scikit-learn | 1.8.0 |
| imbalanced-learn | 0.14.2 |
| Matplotlib | 3.10.8 |
| PyTorch | 2.6.0 |

## License

A project license is pending confirmation of code ownership and release permissions. The TORGO corpus and derived artifacts are subject to their applicable terms.
