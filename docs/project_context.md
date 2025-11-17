# Sign Language Audio Interpreter – Demo (Fall 2025)

## Project Overview
This demo focuses on recognizing **American Sign Language (ASL) letters (A–Z)** and **numbers (0–9)** from images, then converting the recognized character into speech output. The goal is to demonstrate a full end-to-end process: **image input → character classification → audio output**.

This is the first stage of a larger project. Future development will extend to **dynamic gestures and full sign phrases** (e.g., using the How2Sign dataset), but this semester’s focus is on **static image-based recognition**.

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
- A **Convolutional Neural Network (CNN)** is trained to classify each image into one of **36 classes** (A–Z + 0–9).
- The model outputs both the predicted class and its confidence score.
- Implemented using **TensorFlow/Keras** or **PyTorch**.

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
- **Python 3.x**
- **TensorFlow / Keras** or **PyTorch** – model training  
- **OpenCV** – image processing and data loading  
- **gTTS / pyttsx3** – text-to-speech conversion  
- **NumPy, Matplotlib** – data manipulation and visualization

---

## Future Expansion
Future iterations will support:
- **Dynamic gestures** (e.g., J, Z, or motion-based signs)
- **Word-level and sentence-level translation**
- **Real-time webcam input and continuous recognition**

---

*Maintained by:*  
**Duy Nguyen • Duong Banh • Haonan Yu**  
Fall 2025 – CPSC Project Demo
