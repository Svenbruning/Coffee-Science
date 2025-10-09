# Wildlands Visitors Prediction — Quick Setup Guide

## 1. Create Virtual Environment
1. Open a terminal in the project folder (`wildlands-forecast/`)
2. Run:
   ```bash
   python3 -m venv venv
   ```
3. Activate the virtual environment:
   - **Linux/macOS:**  
     ```bash
     source venv/bin/activate
     ```
   - **Windows PowerShell:**  
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
4. Install the required packages:
   ```bash
   pip install flask flask-cors pandas scikit-learn
   ```

---

## 2. Train the Model
1. Go to the ML folder:
   ```bash
   cd ml_model
   ```
2. Train the model:
   ```bash
   python train_model.py
   ```
3. ✅ Output:  
   ```
   Model trained and saved as model.pkl
   ```
4. Check that **model.pkl** was created in the same folder.

---

## 3. Start the Backend
1. Go to the backend folder:
   ```bash
   cd ../backend
   ```
2. Start the Flask server:
   ```bash
   python app.py
   ```
3. ✅ Output:  
   ```
   * Running on http://127.0.0.1:5000
   ```
4. Keep this window open!

---

## 4. Open the Frontend
1. Open a second terminal (keep the backend running!)
2. Go to the frontend folder:
   ```bash
   cd frontend
   ```
3. Start a simple web server:
   ```bash
   python -m http.server 8080
   ```
4. ✅ Output:  
   ```
   Serving HTTP on 0.0.0.0 port 8080 (http://0.0.0.0:8080/)
   ```
5. Open in your browser:  
   👉 http://127.0.0.1:8080
