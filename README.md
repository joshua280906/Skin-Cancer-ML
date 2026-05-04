# 🧠 Skin Cancer Detection using Deep Learning

This project is a deep learning-based system that classifies skin lesion images as **benign** or **malignant**. The goal is to build a simple and effective pipeline that demonstrates how AI can assist in early skin cancer detection.

---

## 🚀 What this project does

- Takes an image of a skin lesion  
- Processes and prepares it for the model  
- Predicts whether it is **benign or malignant**  

---

## 🛠️ Tech Stack

- Python  
- PyTorch / TensorFlow  
- OpenCV  
- NumPy, Matplotlib  

---

## 📂 Project Structure

Skin-Cancer-ML/
│── data/               # Dataset (not included in repo)
│── model/              # Saved trained models
│── src/
│   ├── train.py        # Training script
│   ├── test.py         # Testing script
│   ├── app.py          # App / interface (optional)
│   └── balance_dataset.py
│── requirements.txt
│── .gitignore
│── README.md

---

## ⚙️ Installation

git clone https://github.com/joshua280906/Skin-Cancer-ML.git  
cd Skin-Cancer-ML  
pip install -r requirements.txt  

---

## ▶️ Usage

### Train the model
python src/train.py

### Test the model
python src/test.py

---

## 📊 Model Details

- Uses transfer learning (EfficientNet)  
- Binary classification (Benign vs Malignant)  
- Includes preprocessing and dataset balancing  

---

## 📌 Future Improvements

- Multi-class classification (different skin cancer types)  
- Web-based frontend (Flask / Streamlit)  
- Model deployment  

---

## ⚠️ Disclaimer

This project is for educational purposes only and should not be used for real medical diagnosis.

---

## 🙌 Author

Jenis Joshua Thomas M
