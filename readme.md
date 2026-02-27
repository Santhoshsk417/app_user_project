# 📊 App User Segmentation Dashboard

This project provides a **customer segmentation dashboard** using **K-Means clustering** to analyze app user behavior. The dashboard is built using **Streamlit**, and the analysis helps in identifying high-value, moderate, low-engagement, and occasional users to improve business decision-making.

---

## 🚀 Features

- **Cluster Distribution Visualization**  
  Shows the number of users in each segment (cluster).

- **Cluster Profiling**  
  Displays the mean behavior of users in each cluster for key metrics like sessions, clicks, and engagement score.

- **Customer-Level Identification**  
  Allows viewing individual users in a selected cluster.

- **Business Insights**  
  Provides actionable suggestions for each user segment.

- **Optional PCA Visualization**  
  Visualizes clusters in a 2D PCA space (can be disabled if not needed).

---

## 📂 Project Structure


app-user-segmentation-ml/
│
├── config/
│   ├── config.yaml
│   └── logging.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data/
│   │   ├── data_load.py
│   │   ├── data_cleaning.py
│   │   ├── data_preprocessing.py
│   │   ├── feature_scaling.py
│
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── cluster_labeling.py
│   │   └── model_registry.py
│
│
│   ├── utils/
│   │   ├── logger.py
│
│   └── visual/
│       └── plots.py
│
├── artifacts/
│   ├── models/
│   └── reports/
│
│
├── app/
│   └── streamlit_app.py
│
├── main.py
├── requirements.txt
└── README.md