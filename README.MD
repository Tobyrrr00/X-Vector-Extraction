# X-Vector Speaker Embedding Extraction

An end-to-end implementation of the X-Vector speaker embedding system ([Snyder et al., 2018](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf)), built from scratch in Python. This project covers the full pipeline — from raw audio ingestion through signal preprocessing, perceptual feature extraction, and deep neural network–based embedding generation using a Time-Delay Neural Network (TDNN) architecture.

Developed as an undergraduate thesis at Eastern Washington University (Winter/Spring 2024).  
📄 [Read the full thesis (PDF)](XVectorThesis.pdf)

---

## Pipeline Overview

```
Raw Audio (.wav)
    │
    ▼
┌──────────────────────┐
│  Audio Preprocessing │
│  • Load & resample   │
│  • Frame (25ms/10ms) │
│  • VAD silence trim  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────┐
│   Feature Extraction     │
│  • Hann windowing        │
│  • STFT → Power spectrum │
│  • 24-bin Mel filterbank │
│  • Log scaling (dB)      │
│  • Mean-variance norm    │
└──────────┬───────────────┘
           │
           ▼
    (24 × T) feature matrix
           │
           ▼
┌──────────────────────────────┐
│   X-Vector TDNN Network      │
│  • 5 frame-level TDNN layers │
│  • Statistics pooling        │
│  • 2 segment-level dense     │
│  • Softmax classification    │
└──────────┬───────────────────┘
           │
           ▼
   512-dim speaker embedding
       (the x-vector)
```

---

## Repository Structure

| File | Description |
|------|-------------|
| `01_audio_exploration.ipynb` | Explores raw audio signals — amplitude, frequency, pitch shifting — to build intuition for how sound is represented digitally. |
| `02_preprocessing.ipynb` | Step-by-step walkthrough of the preprocessing pipeline: loading audio, framing, VAD filtering, and Mel filterbank extraction on a single file. |
| `03_feature_extraction.ipynb` | Focused deep-dive into STFT, Mel spectrogram construction, log scaling, and sliding-window normalization with visualizations. |
| `04_full_pipeline.ipynb` | **Main pipeline.** Processes the full dataset, defines the TDNN model in TensorFlow/Keras, trains for speaker classification, and extracts 512-dim x-vector embeddings. |
| `XVectorThesis.pdf` | Full thesis paper covering theory, math, and implementation details. |

---

## Model Architecture

Implemented in TensorFlow/Keras following the architecture from Snyder et al. (2018):

| Layer | Context | Total Context | Input × Output |
|-------|---------|---------------|----------------|
| TDNN 1 | {t−2, t−1, t, t+1, t+2} | 5 frames | 120 × 512 |
| TDNN 2 | {t−2, t, t+2} | 9 frames | 1536 × 512 |
| TDNN 3 | {t−3, t, t+3} | 15 frames | 1536 × 512 |
| TDNN 4 | {t} | 15 frames | 512 × 512 |
| TDNN 5 | {t} | 15 frames | 512 × 1500 |
| Stats Pooling | [0, T] | T | 1500 → 3000 |
| Segment 6 | — | T | 3000 × 512 |
| Segment 7 | — | T | 512 × 512 |
| Softmax | — | T | 512 × N speakers |

The **x-vector embedding** is extracted from the output of Segment Layer 6 — a 512-dimensional fixed-length vector that encodes speaker identity regardless of utterance length.

---

## Dataset

Trained and evaluated on the [RAVDESS](https://zenodo.org/record/1188976) (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset:
- **24 speakers** (12 male, 12 female)
- **1,440 speech utterances** across various emotional expressions
- Audio sampled at 22,050 Hz

---

## Key Technical Details

- **Framing:** 25 ms frames with 10 ms hop (551 samples / 220 samples at 22,050 Hz)
- **VAD:** Energy-based voice activity detection at −40 dB threshold; retains all frames between first and last voiced segment to preserve natural pauses
- **Features:** 24-bin log-Mel filterbank energies (20–7,600 Hz), normalized via 3-second sliding-window mean subtraction
- **Windowing:** Hann window to reduce spectral leakage before STFT
- **Training:** Adam optimizer (lr=1e-3), sparse categorical cross-entropy, batch size 32, 3-second random crops for training segments
- **Framework:** TensorFlow 2.x / Keras

---

## Getting Started

### Requirements

```
numpy
pandas
librosa
matplotlib
seaborn
tensorflow>=2.0
scikit-learn
```

### Running the Pipeline

The exploratory notebooks (`01`–`03`) can be run independently to visualize individual pipeline stages.

1. Place audio files in `0audio/` organized as `0audio/actor_XX/*.wav`
2. Run `04_full_pipeline.ipynb` (or `full_program.ipynb`) end-to-end:
   - Processes all audio files through VAD and feature extraction
   - Trains the TDNN for speaker classification
   - Extracts 512-dim x-vector embeddings for each utterance



---

## References

- Snyder, D., Garcia-Romero, D., Sell, G., Povey, D., & Khudanpur, S. (2018). *X-vectors: Robust DNN Embeddings for Speaker Recognition.* ICASSP 2018.
- Snyder, D., Garcia-Romero, D., Povey, D., & Khudanpur, S. (2017). *Deep Neural Network Embeddings for Text-Independent Speaker Verification.* Interspeech 2017.
- Livingstone, S. R. & Russo, F. A. (2018). *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).* Zenodo.

**And I will always give a special thanks to:**
- Backstrom, T. et. al. (2022), *Introduction to Speech Processing - Introductionto Speech Processing 2dn edition.* https://speechprocessingbook.aalto.fi. 10.5281/zenodo.6821775