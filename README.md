Project Link: https://f1-prediction-project.streamlit.app/
Last Data Update: Hungarian GP 2025 race

# My F1 Race Predictor

A Formula 1 race prediction app that uses machine learning on 2023-2025 F1 data to predict the podium finishers of the next race.

### 1. Data Collection

- Used OpenF1 API to collect 3 years of F1 data (2023-2025).
- Gathered 1,159 driver-race combinations across 45+ races.
- Extracted qualifying results, race outcomes, weather, and driver/team info.

### 2. Feature Engineering

- Created 47 predictive features from raw data.
- Built historical metrics: driver career stats, recent form, circuit-specific performance.
- Added team performance indicators and weather conditions.
- Handled new drivers and mid-season team changes.

### 3. Model Development

- Trained multiple algorithms (Random Forest, Gradient Boosting, Logistic Regression).
- Used time-based split: 2023-2024 for training, 2025 for validation.
- Achieved 91% accuracy and 0.967 AUC score on real future races.
- Identified qualifying position as the strongest predictor (22% importance).

### 4. Dashboard Creation

- Built interactive Streamlit web application.
- Grid-based qualifying setup with driver dropdowns.
- Real-time predictions with confidence scores.
- Weather condition controls and probability visualizations.

### 5. Cloud Deployment

- Deployed to Streamlit Community Cloud.
- Handled version compatibility issues for cloud environment.
- Auto-retraining system for maximum compatibility.
  
The project is live at: https://f1-prediction-project.streamlit.app/
