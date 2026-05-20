# Deep Learning Approach to Spoken Digit Recognition (Edge Optimized)

This repository contains an end-to-end implementation of an isolated, ultra-lightweight **Automatic Speech Recognition (ASR) system** built with PyTorch to classify spoken numerical digits (0–9). 

## ⚡ Technical Architecture
* **Spectrogram Feature Processing:** Spoken frequencies are formatted as 2D Mel-Frequency Cepstral Coefficients (MFCC) feature matrices ($40 \times 50$), transforming temporary audio oscillations into stable spatial images.
* **Edge-Ready 2D Convolutional Core:** Implements a localized deep network using Batch Normalization and specialized Dropouts to secure stable inference footprints under restricted processing limits (e.g., mobile devices or embedded microcontrollers).
* **Self-Contained Pipeline Environment:** Contains a built-in synthetic spectrogram generation layer to make the entire optimization ecosystem testable instantly without setting up complex system-level OS sound libraries.

## 🚀 Getting Started

### 1. Build Virtual Environment dependencies
```bash
pip install -r requirements.txt