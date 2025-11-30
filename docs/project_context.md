# ASL Audio Interpreter - Project Overview

## Current Status
**Model Accuracy: 99.74%** (Achieved November 29, 2025)  
**Production Ready** - Ready for real-time deployment

## Project Description
This project recognizes **American Sign Language (ASL) letters (A–Z)** and **numbers (0–9)** from images using deep learning, then converts the recognized character into speech output. The goal is to demonstrate a full end-to-end process: **image input → character classification → audio output**.

This is the first stage of a larger project. Future development will extend to **dynamic gestures and full sign phrases** (e.g., using the How2Sign dataset), but the current focus is on **static image-based recognition**.

---

## How the Demo Works

### 1. **Input**
- The system takes an image containing an ASL hand sign for any **letter (A–Z)** or **number (0–9)**.
- Images are sourced from a publicly available dataset ([ASL Dataset on Kaggle](https://www.kaggle.com/datasets/ayuraj/asl-dataset)).

### 2. **Preprocessing**
- Images are resized, normalized, and labeled for training.
- The dataset is split into training, validation, and test sets.
- Data augmentation (e.g., rotation, zoom, flipping) may be applied to improve model robustness.

### 3. **Model**
- A **Convolutional Neural Network (CNN)** with 4 blocks and 19.25M parameters classifies each image into one of **36 classes** (A–Z + 0–9).
- The model achieves **99.74% test accuracy** with 35/36 classes at 100%.
- Architecture: Conv(64)→Conv(128)→Conv(256)→Conv(512)→Dense(512)→Dense(256)→Output(36)
- Implemented using **TensorFlow/Keras** with batch normalization and dropout for regularization.

### 4. **Text-to-Speech (TTS)**
- After classification, the recognized letter or number is passed to a TTS engine.
- **gTTS** or **pyttsx3** converts the output into audible speech.

### 5. **Output**
- The system displays the predicted character and plays the corresponding audio.  
  Example:  
  → Input: image of the hand sign for “3”  
  → Output: “Predicted: 3” and the audio says “Three”

---

## Tools & Libraries
- **Python 3.8+**
- **TensorFlow 2.15+ / Keras 3.0+** – deep learning framework
- **OpenCV 4.8+** – image processing and webcam capture
- **scikit-learn 1.3+** – data splitting, metrics, class weights
- **gTTS / pyttsx3** – text-to-speech conversion  
- **NumPy, Pandas, Matplotlib, Seaborn** – data manipulation and visualization

## Model Performance
- **Test Accuracy:** 99.74%
- **Training Time:** ~60-90 minutes (150 epochs with early stopping)
- **Model Size:** 73.44 MB (19.25M parameters)
- **Input Resolution:** 128×128 RGB images
- **Classes:** 36 (A-Z, 0-9)
- **Dataset:** 2,515 images (70% train / 15% val / 15% test)

---

## Future Expansion
Future iterations will support:
- **Dynamic gestures** (e.g., J, Z, or motion-based signs)
- **Word-level and sentence-level translation**
- **Real-time webcam input and continuous recognition**

---

## Documentation
- **Training Results:** See `TRAINING_RESULTS.md` for complete performance history
- **V2 Improvements:** See `MODEL_IMPROVEMENTS_V2.md` for implementation details
- **Project README:** See `../README.md` for setup and usage instructions

---

*Maintained by:*  
**Duy Nguyen • Duong Banh • Haonan Yu**  
Fall 2025 – CPSC Capstone Project

**Model Version:** V2 (Enhanced Architecture)  
**Last Updated:** November 29, 2025
