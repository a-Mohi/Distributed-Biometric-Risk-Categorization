# Distributed Biometric Risk Categorization 

This repository implements a **Distributed Healthcare Monitoring System** designed to categorize patient health risks using bio-sensor data. Leveraging a large-scale dataset of **200,020 records**, the system simulates a distributed environment across hospital wards to identify critical medical cases while strictly prioritizing patient safety through the minimization of **False Negatives**.

---

##  Project Overview

* **Architecture:** Distributed Node Processing (Simulated Wards A, B, and C).
* **Data Scale:** 200,000+ Human Vital Sign & Clinical Metric records.
* **Primary Objective:** Identification of "High Risk" patients with clinical-grade accuracy.
* **Core Methodology:** Safety-weighted Random Forest Classification with a focus on **Recall (Sensitivity)** optimization.

---

##  Project Workflow

### Day 1: Distributed Data Splitting
Implemented data sovereignty and privacy principles by partitioning the global dataset into three independent, ward-level nodes. This ensures that data is processed locally, simulating a real-world secure hospital network.

### Day 2: Local Preprocessing & Training
Each node performed autonomous local cleaning and feature engineering. Models were trained using **Cost-Sensitive Learning**, applying heavy weights to critical cases to ensure the system is highly sensitive to medical emergencies.

### Day 3: Global Aggregation
Developed a central monitoring logic that aggregates high-priority alerts from distributed nodes. This allows for a global "Hospital Dashboard" view without requiring the movement of raw, sensitive patient data across the network.

### Day 4: Performance Evaluation
Validated the system using **ROC-AUC Curves** and **Confusion Matrices**. The evaluation focused on the "Safety First" metric, ensuring that the rate of missed critical cases (False Negatives) was strictly minimized.

---

##  Contributors & Team Structure

###  Model Development Team
*Focused on Random Forest architecture, safety fine-tuning, and algorithmic optimization.*
* **Ammar Mohiuddin** (President)
* **Anand Murthy R** (Secretary)
* **Mohammed Zaid Ali**
* **Athish Prajwal GR** (Deputy Secretary)

###  EDA Team (Exploratory Data Analysis)
*Responsible for feature correlation, physiological threshold identification, and vital sign distribution analysis.*
* **Sindhu M**
* **Kiran Reddy T R**
* **Sonali P Bhasme**

###  Data Cleaning & Preprocessing Team
*Responsible for local node normalization, handling missing values, and engineering metrics such as BMI and HRV.*
* **Suha Maria** (Vice President)
* **Deepshikha Vishwakarma**
* **Aradhana Prajapati**

###  Evaluation Metrics Team
*Focused on ROC-AUC generation, performance validation, and ensuring the reduction of False Negatives.*
* **Swati Jadhav**
* **GVS Manashwi Roy**
* **Wafaa S M Abukallousah**
* **Maulya Naik**

---

##  Key Results

* **Sensitivity (Recall):** Successfully optimized to >99%, ensuring near-zero missed critical cases.
* **ROC AUC:** Demonstrated exceptional separability between health risk categories.
* **Distributed Scalability:** Confirmed that local nodes can maintain high predictive power while operating on independent datasets.
