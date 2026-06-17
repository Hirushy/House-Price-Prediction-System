# AI Property Valuation System

A sophisticated, full-stack web application designed to predict real estate prices using machine learning. This system offers users a premium, interactive interface to input property features and instantly receive a market value estimate, accompanied by confidence scores and a breakdown of the key factors driving the valuation.

## Features
*   **Machine Learning Backend:** Powered by an XGBoost model built with Scikit-Learn pipelines, ensuring highly accurate and robust price predictions based on a rich dataset.
*   **Interactive Analytics:** Features dynamic charts that visually break down the "Market Drivers" (Feature Importance), helping users understand exactly which attributes (e.g., Province, District, Has Garden) influence their property's value.
*   **Premium Glassmorphism UI:** A sleek, modern, and fully responsive frontend built with React and Framer Motion. It includes a stunning, interactive 3D background powered by Three.js and `@react-three/fiber`.
*   **Advanced Configuration:** Users can toggle "Advanced Settings" to fine-tune their predictions by entering specific details like Property Age, Distance to Colombo, Parking Spaces, and more.

##  Tech Stack
*   **Frontend:** React.js, Vite, Framer Motion (for animations), Three.js / React Three Fiber (for 3D models), Recharts (for analytics).
*   **Backend:** Python, FastAPI, Pandas, Scikit-Learn, XGBoost.

##  How to Run Locally

### 1. Start the Backend (FastAPI)
```bash
cd backend
# Activate your virtual environment (Windows)
.\venv\Scripts\activate
# Start the server
uvicorn app:app --reload
```
*The backend server will run on `http://127.0.0.1:8000`*

### 2. Start the Frontend (React/Vite)
Open a new terminal window:
```bash
cd frontend
npm run dev
```
*The frontend application will be available at `http://localhost:5173`*
