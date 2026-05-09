# 🧠 Oral Cancer Detection using AI

### Early Diagnosis through Deep Learning & Image Analysis

---

## 📌 Abstract

Oral cancer is among the most critical health concerns globally, where **early detection plays a vital role in increasing survival rates**. This project presents an **AI-driven system** that leverages deep learning techniques to analyze oral cavity images and detect potential cancerous conditions.

The system integrates **computer vision, machine learning, and full-stack development** to provide a seamless pipeline from image input to prediction output.

---

## 🎯 Objectives

* Develop a reliable AI model for oral cancer detection
* Automate early diagnosis using image classification
* Build an end-to-end system (frontend + backend + ML model)
* Ensure scalability and real-time usability

---

## 🚀 Key Features

* 🔍 **Image-Based Detection**
  Upload oral cavity images for instant analysis

* 🤖 **Deep Learning Model**
  Uses **ResNet50V2** for high-performance classification

* 📊 **Accurate Predictions**
  Binary classification: *Cancerous / Non-Cancerous*

* 🌐 **Full-Stack Integration**
  Smooth communication between frontend UI and backend APIs

* 🧪 **Testing Support**
  Includes API and backend testing scripts

* 📁 **Structured Dataset Pipeline**
  Organized data handling for training and validation

---

## 🏗️ System Architecture

```
User (Frontend)
      ↓
Image Upload (UI)
      ↓
Backend API (Flask/FastAPI)
      ↓
Preprocessing (OpenCV, NumPy)
      ↓
Deep Learning Model (ResNet50V2)
      ↓
Prediction Output
      ↓
Display Result (Frontend)
```

---

## 🛠️ Tech Stack

### 👨‍💻 Frontend

* HTML
* CSS
* JavaScript

### ⚙️ Backend

* Python
* Flask / FastAPI

### 🤖 Machine Learning

* TensorFlow / Keras
* ResNet50V2 (Transfer Learning)

### 📊 Data Processing

* OpenCV
* NumPy
* Pandas

### 🔧 Tools & Platforms

* Git & GitHub
* VS Code
* Postman (API testing)

---

## 📂 Project Structure

```
Oral-Cancer-AI/
│── backend/              # API & server logic
│── frontend/             # User interface
│── data/                 # Dataset (if included)
│── train/                # Training scripts & models
│── static/               # Static assets
│── test_images/          # Sample images for testing
│── test_api.py           # API testing
│── test_backend.py       # Backend testing
│── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Ashutosh-Chaudhari/Oral_Cancer_Detection.git
cd Oral_Cancer_Detection
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Start Backend Server

```bash
cd backend
python app.py
```

> **Note:** The `data/` folder ships with a small sample (5 images per class) for demonstration. The full validation set is not included in the repository.

### Run Frontend

* Open `index.html` in browser
  OR
* Use Live Server (VS Code)

---

## 🧠 Model Details

* **Model:** ResNet50V2 (Transfer Learning)
* **Task:** Binary Classification
* **Input:** Oral cavity images
* **Output:** Cancer / Non-Cancer

### Training Approach

* Data preprocessing (resizing, normalization)
* Data augmentation for better generalization
* Fine-tuning pre-trained layers
* Validation-based performance monitoring

---

## 📊 Performance Metrics *(Update if available)*

| Metric    | Value |
| --------- | ----- |
| Accuracy  | XX%   |
| Precision | XX%   |
| Recall    | XX%   |
| F1 Score  | XX%   |

---

## 🧪 Testing

Run test scripts to verify functionality:

```bash
python test_api.py
python test_backend.py
```

---

## 📸 Screenshots

*Add UI screenshots here for better presentation*

---

## ⚠️ Limitations

* Model performance depends on dataset quality
* Not a replacement for professional medical diagnosis
* Requires further clinical validation

---

## 🔮 Future Improvements

* 🔹 Multi-class classification (different cancer stages)
* 🔹 Mobile app integration
* 🔹 Real-time camera detection
* 🔹 Cloud deployment (AWS / GCP / Azure)
* 🔹 Federated Learning for privacy-preserving training

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Ashutosh Chaudhari**
B.Tech Computer Engineering Student

---

## ⭐ Acknowledgements

* Open-source ML libraries
* Medical imaging research contributions
* Community support

---
