Here's a properly formatted **`README.md`** file that is easy to read and navigate:

---

# 🚀 **Startup Project - AI Legal Compliance Review System**

This project is an **AI-powered legal document review system** designed to streamline compliance checks for financial institutions.

It consists of:

- **Backend**: A FastAPI server for processing legal documents.
- **Frontend**: A React-based web interface for users to upload and review documents.

---

## 📂 **Project Structure**

```
STARTUPPROJECT/
│── backend/               # FastAPI backend
│   ├── __pycache__/       # Compiled Python files
│   ├── uploads/           # Directory for uploaded documents
│   ├── venv/              # Virtual environment (ignored by Git)
│   ├── main.py            # FastAPI entry point
│   ├── requirements.txt   # Backend dependencies
│
│── frontend/              # React frontend
│   ├── node_modules/      # Installed dependencies (ignored by Git)
│   ├── public/            # Static assets
│   ├── src/               # Main source code
│   │   ├── components/    # Reusable UI components
│   │   ├── App.js         # Main application file
│   │   ├── index.js       # React entry point
│   │   ├── App.css        # Global styles
│   │   ├── index.css      # Base styles
│   ├── package.json       # Frontend dependencies
│   ├── package-lock.json  # Package lock file
│
│── .gitignore             # Ignore unnecessary files (venv, node_modules)
│── README.md              # Project documentation
```

---

## 📥 **Installation Guide**

### 🔹 **Prerequisites**

Before setting up the project, ensure you have installed:

- **Python 3.9+** → [Download](https://www.python.org/downloads/)
- **Node.js 18+** → [Download](https://nodejs.org/)
- **npm** (Node Package Manager) → [Guide](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)

---

## ⚙️ **Backend Setup (FastAPI)**

### **1️⃣ Clone the Repository**

```bash
git clone https://github.com/your-username/startup-project.git
cd startup-project
```

### **2️⃣ Navigate to the Backend**

```bash
cd backend
```

### **3️⃣ Create a Virtual Environment**

```bash
python -m venv venv
```

### **4️⃣ Activate the Virtual Environment**

- **Windows (CMD/PowerShell):**
  ```bash
  venv\Scripts\activate
  ```
- **Mac/Linux:**
  ```bash
  source venv/bin/activate
  ```

### **5️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

### **6️⃣ Run the Backend**

```bash
uvicorn main:app --reload
```

✅ Your **FastAPI backend** will now be running on:  
🔗 **`http://127.0.0.1:8000`**

---

## 💻 **Frontend Setup (React)**

### **1️⃣ Navigate to the Frontend**

```bash
cd ../frontend
```

### **2️⃣ Install Dependencies**

```bash
npm install
```

### **3️⃣ Start the Frontend**

```bash
npm start
```

✅ Your **React app** will now be running on:  
🔗 **`http://localhost:3000`**

---

## 🚀 **Running the Full Project**

To run **both backend and frontend simultaneously**, open **two terminals**:

### **Terminal 1 (Backend)**

```bash
cd backend
source venv/bin/activate   # Windows: venv\Scripts\activate
uvicorn main:app --reload
```

### **Terminal 2 (Frontend)**

```bash
cd frontend
npm start
```

---

## 🛠 **Environment Variables**

For security, environment variables should be stored in a `.env` file.

### **Backend (`.env`)**

```ini
DATABASE_URL=postgresql://user:password@localhost:5432/mydb
SECRET_KEY=your_secret_key
```

### **Frontend (`.env`)**

```ini
REACT_APP_BACKEND_URL=http://localhost:8000
```

---

## 🔗 **API Endpoints**

| Endpoint   | Method | Description                              |
| ---------- | ------ | ---------------------------------------- |
| `/`        | GET    | Root route                               |
| `/upload`  | POST   | Upload a legal document                  |
| `/analyze` | POST   | Analyze a document for compliance issues |

---

## 📌 **Best Practices**

✅ **Use Virtual Environments (`venv`)** for Python  
✅ **Ignore `node_modules/` & `venv/` in Git**  
✅ **Follow Git Branching Strategy** (`feature-branch` → `main`)  
✅ **Run `npm install` and `pip install -r requirements.txt` before development**

---

## **🔹 Setting Up Google Gemini API Access**

To integrate **Gemini AI** into your project, follow these steps:

### **1️⃣ Install Google Cloud SDK (`gcloud`) using Homebrew (macOS)**

Run the following command in your **terminal** to install the Google Cloud SDK:

```bash
brew install --cask google-cloud-sdk
```

After installation, **restart your terminal** and verify the installation:

```bash
gcloud --version
```

---

### **2️⃣ Authenticate with Google Cloud**

Run this command to log in:

```bash
gcloud auth application-default login
```

This will open a browser window where you need to authenticate with your **Google Cloud account**.

---

### **3️⃣ Set the Correct Project**

Find your **Google Cloud Project ID**:

```bash
gcloud projects list
```

Set the active project using:

```bash
gcloud config set project YOUR_PROJECT_ID
```

Replace **`YOUR_PROJECT_ID`** with your actual project ID.

Verify the project was set correctly:

```bash
gcloud config list
```

---

### **5️⃣ Get Your Gemini API Key**

1. Go to [Google AI Studio](https://aistudio.google.com/) and **sign in**.
2. Click on **"Get API Key"**.
3. Copy the API key.

---

### **6️⃣ Store the API Key in `.env` File**

For security, **never hardcode your API key in the source code**. Instead, add it to your **backend `.env` file**:

#### **Backend (`.env`)**

```ini
GEMINI_API_KEY=your_google_gemini_api_key
```

🔹 **Replace** `your_google_gemini_api_key` with the actual **Gemini API Key**.

---
