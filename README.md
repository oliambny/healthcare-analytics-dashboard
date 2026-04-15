# Real-Time Healthcare Analytics Dashboard (Synthetic, HIPAA-Safe)

This project is a **real-time healthcare analytics dashboard** built with **Streamlit**, using **fully synthetic, HIPAA-safe data** to simulate:

- Patient-level time-series vitals  
- Risk scoring using a machine learning model  
- Anomaly detection for unusual patterns  

It is designed to showcase skills in:

- Data engineering & synthetic data generation  
- Machine learning for risk prediction & anomaly detection  
- Interactive analytics dashboards for healthcare operations  

## Features

- Synthetic patient-day dataset with:
  - Heart rate, blood pressure, temperature, respiratory rate
  - Derived risk score and binary high-risk label
- RandomForest-based risk model
- IsolationForest-based anomaly detection
- Interactive dashboard:
  - Global KPIs for a selected day
  - Risk distribution visualization
  - Anomaly vs risk scatter plot
  - Patient-level time-series views

## Tech Stack

- **Python**
- **Streamlit** for the dashboard
- **Pandas / NumPy** for data handling
- **scikit-learn** for ML models
- **Plotly** for interactive charts
