# 🧠 Real-Time Sign Language Recognition

This project uses a deep learning model with OpenCV and CVZone to recognize hand gestures for sign language in real-time using a webcam.

## 📂 Project Files
- `check.py`: Python script to capture webcam input and predict signs.
- `labels.txt`: Contains gesture class labels.
- `requirements.txt`: List of required libraries.

## ⚙️ Requirements
Install dependencies using:

```bash
pip install -r requirements.txt
```

## 🚀 How to Run

1. Place the `keras_model.h5` and `labels.txt` in the same folder as `check.py`.
2. Run the code:
```bash
python check.py
```
3. Use hand gestures in front of the webcam to see predictions.

## 📥 Model File
The `.h5` model file is too large to include.  
👉 [Click here to download keras_model.h5](https://drive.google.com/file/d/1q898QpQZNjlJHvV9eOsnn7ujcmdWDk4M/view?usp=sharing)

After downloading, place it in the same folder.

## 🧠 Labels Used
```
0 Stop
1 Iloveyou
2 Pain
3 Hurts
```

## 👩‍💻 Author
Shalini D.  
Real-Time Sign Language Recognition System
