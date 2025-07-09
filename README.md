
# 🫁 Pneumonia Detection from Chest X-Rays using CNN

This project implements a Convolutional Neural Network (CNN) to detect **Pneumonia** from **Chest X-ray** images. The model is trained on a publicly available dataset and achieves high accuracy in classifying images as either **Normal** or **Pneumonia**.

---

## 📌 Table of Contents

- [📂 Project Structure](#-project-structure)
- [📁 Dataset](#-dataset)
- [🚀 Features](#-features)
- [🧠 Model Architecture](#-model-architecture)
- [📊 Results](#-results)
- [📈 Training Visualization](#-training-visualization)
- [⚙️ How to Use](#️-how-to-use)
- [🧾 Requirements](#-requirements)
- [🤝 Contributing](#-contributing)
- [📃 License](#-license)

---

## 📂 Project Structure

```
├── chest_x_ray.py
├── cnn_model.keras
├── x-ray-photo-image-chest-Normal.jpg
├── x-ray-photo-image-chest-Pneumonia.jpg
├── requirements.txt
└── README.md
```

---

## 📁 Dataset

This project uses the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset provided by **Paul Timothy Mooney** on Kaggle.

### 📦 Download Instructions

1. Download the dataset from Kaggle:  
   👉 https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

2. Extract the dataset and structure it like this:

```
/chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

---

## 🚀 Features

- Loads and preprocesses grayscale X-ray images.
- Builds a CNN with dropout layers for regularization.
- Trains on labeled data with real-time validation.
- Evaluates model performance on test data.
- Predicts external images for real-world testing.
- Visualizes correct and incorrect classifications.

---

## 🧠 Model Architecture

- **Input:** Grayscale images resized to 256x256
- **Layers:**
  - 4x `Conv2D → ReLU → MaxPool → Dropout`
  - Flatten + Dense(256)
  - Output: Dense(1, activation=`sigmoid`)
- **Optimizer:** Adam (`lr = 1e-5`)
- **Loss Function:** Binary Crossentropy
- **EarlyStopping** callback to prevent overfitting

---

## 📊 Results

```bash
Test Loss: 0.22
Test Accuracy: 94.6%
```

---

## 📈 Training Visualization

### 🔵 Accuracy Plot

![Accuracy](https://user-images.githubusercontent.com/Mayankchauhan008/accuracy-plot.png)

### 🔴 Loss Plot

![Loss](https://user-images.githubusercontent.com/Mayankchauhan008/loss-plot.png)

> Replace these placeholder URLs with actual image links from your repo.

---

## ⚙️ How to Use

### 🔹 Clone the Repository

```bash
git clone https://github.com/Mayankchauhan008/chest-xray-pneumonia-detector.git
cd chest-xray-pneumonia-detector
```

### 🔹 Install Dependencies

```bash
pip install -r requirements.txt
```

### 🔹 Train the Model

```bash
python chest_x_ray.py
```

### 🔹 Predict on New Image

```python
model = tf.keras.models.load_model("cnn_model.keras")

def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (256, 256))
    return new_array.reshape(-1, 256, 256, 1)

labels = ["NORMAL", "PNEUMONIA"]
prediction = model.predict([prepare("your_image.jpg")])
print(labels[int(prediction[0])])
```

---

## 🧾 Requirements

```txt
tensorflow>=2.9.0
numpy
pandas
opencv-python
matplotlib
seaborn
Pillow
```

To install:

```bash
pip install -r requirements.txt
```

---

## 🖼️ Sample Prediction Images

| Normal | Pneumonia |
|--------|-----------|
| ![Normal](x-ray-photo-image-chest-Normal.jpg) | ![Pneumonia](x-ray-photo-image-chest-Pneumonia.jpg) |

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome!  
Feel free to open issues or submit pull requests.

---

## 📃 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for more details.

---

## 🙏 Acknowledgments

- [Kaggle - Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- TensorFlow, OpenCV, and Matplotlib contributors

---
