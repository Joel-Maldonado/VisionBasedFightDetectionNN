# Vision-Based Fight Detection for Surveillance Cameras

A deep learning model built using TensorFlow for detecting violent scenes in surveillance camera footage. The model analyzes video input and provides:

- Clip-level violence score (0–100%)
- Per-frame bounding box localizing the suspected altercation

## Environment

- Python 3.9+ is recommended (project developed with Python 3.9).

## Model Weights

This repository includes training and inference scripts, example data, and plots. Pretrained model weights and large training artifacts are not included in the repository. To reproduce results, download the dataset yourself, use the provided scripts to generate TFRecords and train models locally, then point the helper scripts to your resulting checkpoints.

## Dataset

This project utilizes the [RWF-2000 dataset](https://arxiv.org/pdf/1911.05913v3) for training. Reach out to the authors for access.

## Project Background

This project was developed as part of the **Programming for Social Good** internship at Saturday Academy, which aims to leverage technology for positive social impact. By creating a model that swiftly detects violent scenes in surveillance footage, this project has the potential to enhance public safety and improve law enforcement response times to incidents.

## Presentation

This project was presented at the ASE Saturday Academy Symposium 2022. You can watch the presentation on YouTube [here](https://www.youtube.com/watch?v=WD-oEW791a0&t=4523s).

## Example

https://github.com/user-attachments/assets/3bdb048e-4e5a-4161-8c52-cc14c4c4896f
