# Distributed Biometric Risk Categorization - Project Report 

## Executive Summary
This project designs an advanced classification system utilizing distributed data principles to categorize patient health risks securely from biometric sensor data. The core strategy implemented is a **hybrid distributed learning system** (Federated Averaging via PyTorch and Centralized Dashboarding via Random Forests), ensuring patient records remain decentralized while collaboratively training a unified global model. A paramount requirement of this system was strictly minimizing false negative classifications for critical health cases.

## 1. Data Preprocessing and Distributed Simulation
To simulate a realistic decentralized health environment, the synthetic patient vital signs dataset (`human_vital_signs_dataset_2024.csv`) was filtered for the relevant biometric features: Heart Rate, Respiratory Rate, Body Temperature, Oxygen Saturation, Blood Pressure (Systolic/Diastolic), Age, Gender, Derived BMI, and Derived HRV.

**Distributed Setup (Data Partitioning)**
The dataset was randomly shuffled to ensure an unbiased distribution and subsequently split into three independent segments, representing three distinct healthcare facilities or nodes (`ward_a`, `ward_b`, `ward_c`).

## 2. Distributed Learning Approaches
The project successfully implemented two advanced architectures:

**1. Secure Federated Averaging (PyTorch Deep Learning)**
- A Neural Network (MLP) acts as the foundational architecture. Each node independently normalizes its local biometric data and trains a localized copy.
- The nodes securely transmit only their **learned model weights/gradients** to a central server. The raw patient history never leaves the individual hospital node, ensuring maximum privacy compliance.

**2. Rule-Based Edge Computing (Random Forest Ensemble)**
- Node-level Random Forest classifiers are trained at the edge. 
- Instead of transmitting weights, each local node analyzes its patients and transmits only **Critical High-Risk Alerts** to a central monitoring dashboard, complete with human-readable diagnostic notes (e.g., *Respiratory Warning*, *Cardiovascular Stress*).

## 3. Minimizing High-Risk False Negatives (Critical Cases)
To address the critical requirement of minimizing False Negatives (missed diagnoses), both algorithms utilized **Cost-Sensitive Learning**:
- High-Risk cases were artificially heavily weighted (e.g., `[1.0, 5.0]` in Neural Network Cross-Entropy and `{0: 1, 1: 100}` in Random Forest). This mathematically exponentially inflates the cost of misdiagnosing a critical individual. 
- Classification probability thresholds were adjusted downwards, ensuring ambiguous cases lean safely towards "High Risk".

## 4. Evaluation Metrics & Data Bias Reality
Both models achieved exceptionally high sensitivity/recall, nearly eliminating False Negatives.

- **ROC Curve Generation & The 1.0 AUC Anomaly**: A visual Receiver Operating Characteristic (ROC) curve plotted a nearly perfect `1.0` AUC for the PyTorch model. 
- **Data Distribution Investigation:** Introspection revealed this was due to the synthetic nature of the dataset. For instance, **zero** "Low Risk" patients had a Heart Rate over 90. Because the dataset exhibits extreme linear separability, the models drew easy decision boundaries.

**Crucial Clinical Safety Insight (Handling Extremes)**
Because machine learning algorithms inherently cannot extrapolate "Out-Of-Distribution" (OOD) data, an extreme/fatal input (e.g., an SpO2 of 0% or Temp of 60°C) that the model has never seen during training is mathematically misunderstood and classified as "Normal Risk". 

**Conclusion:** 
Real-world deployment of this system strictly requires a **Clinical Hybrid Approach**, where hardcoded medical thresholds ("If SpO2 < 85: Immediate Alert") intercept the data *before* it reaches the predictive ML model.
