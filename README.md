# Acoustic Gatekeeper

A lightweight Machine Learning pipeline designed to pre-filter acoustic data for drone tracking arrays. This 1D CNN gatekeeper processes audio arrays to identify drone rotor frequencies while aggressively filtering out environmental background noise, preventing matrix explosion in downstream Time Difference of Arrival (TDOA) solvers.

## Project Scope & Legal Disclaimer
This repository is strictly for **educational, experimental, and learning purposes**. It is not currently intended for commercial deployment. 

## Dataset Attributions
The machine learning models in this repository were trained using data from the following publicly available acoustic datasets:
* **ITU ARIS Lab. Acoustic Drone Dataset:** Muhacıroğlu, İ. M. and Akgül, T. (2026). Published under CC-BY-4.0. Contains continuous outdoor drone flight intervals and background environmental noise.[cite: 3, 4]
* **Saraleemadi Drone Audio Dataset:** Binary and multiclass (Parrot Bebop / Mambo) acoustic flight signatures.
* **Gautam Acoustic Drone Dataset (Kaggle):** Multi-distance drone field recordings and ambient environmental audio.

## Technical Architecture
* **Input Dimensionality:** `[16000, 1]` (Exactly 1.000s duration, 16 kHz, Mono).
* **Feature Extraction:** 40-band Mel-Frequency Cepstral Coefficients (MFCCs).
* **Model:** 1D Convolutional Neural Network deployed via TensorFlow Lite (`.tflite`) for sub-millisecond edge inference.
