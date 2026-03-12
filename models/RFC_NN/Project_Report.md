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

## 4. Model Performance Comparison
After retraining the tree-based models with aggressive data augmentation and weight adjustment, the overall system was evaluated. Logistic Regression was removed due to its inability to reliably flag high-risk clinical cases.

| Model | Accuracy | Recall (High-Risk) | False Negatives | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest (RFC)** | 0.9999 | 1.0000 | 0 | Highly reliable edge-case detection |
| **XGBoost** | 0.9978 | 0.9984 | 34 | Scalability and rapid training |
| **Neural Network (MLP)** | 0.9900 | 1.0000 | 0 | Distributed/Federated Learning |

**Key Findings:**
1. **Recall Supremacy**: Random Forest and the Neural Network reached perfect or near-perfect recall on safety-critical cases after oversampling.
2. **Precision Trade-off**: Aggressive safety tuning increased False Positives (Low Risk flagged as High Risk), which is the preferred clinical tradeoff.

## 5. Clinical Safety & Data Bias Reality
Introspection revealed that the extreme linear separability of the synthetic dataset led to "perfect" metrics ($1.0$ AUC), which may not persist in real clinical data.

**Crucial Clinical Safety Insight:**
Because machine learning algorithms inherently cannot extrapolate "Out-Of-Distribution" (OOD) data, an extreme/fatal input that the model has never seen during training could be misclassified. 

## 6. Final Recommendations
1. **Hybrid Clinical Safety**: Real-world deployment MUST utilize hardcoded medical thresholds ("Hard Stops") to catch obvious vitals crises before the ML model processes them.
2. **Distributed Resilience**: The federated approach for the Neural Network ensures privacy while the Random Forest's rule-based nature provides interpretability at the facility level.

**Conclusion:**
The project successfully demonstrates a multi-tier, distributed risk categorization system that prioritizes patient safety through aggressive data tuning and hybrid architectural safeguards.

