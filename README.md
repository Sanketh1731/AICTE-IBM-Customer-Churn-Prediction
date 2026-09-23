# Customer Churn Prediction & Retention BI Dashboard

[![AICTE | IBM SkillsBuild Internship](https://img.shields.io/badge/AICTE%20%7C%20IBM%20SkillsBuild-Internship%202026-0284c7.svg)](https://skills.yourlearning.ibm.com/)
[![Partner - BharatCares](https://img.shields.io/badge/Partner-BharatCares%20(SMEC%20Trust)-10b981.svg)](https://bharatcares.org/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)

---

## 📌 Project Overview
This project was developed as part of the **AICTE | IBM SkillsBuild Data Analytics with AI Academic Internship Program 2026**, conducted by **BharatCares (SMEC Trust)** in association with **AICTE** and **IBM**.

The goal of this project is to bridge the gap between passive reporting and actionable business decisions by implementing an **Enterprise Customer Churn Prediction & Retention Business Intelligence (BI) Dashboard**. Built on a complete **5-tier Business Intelligence hierarchy**, the platform enables stakeholders to monitor churn KPIs in real time, uncover root cause drivers of customer attrition, and leverage machine learning classifiers to predict individual churn risk with personalized retention action plans.

- **Author / Student Name:** Sanketh
- **Student Email:** sankethbkr2005@gmail.com
- **Internship Track:** Data Analytics with AI: Foundation to Implementation

---

## 📂 Dataset Source & Information
- **Dataset Name:** Telco Customer Churn Dataset
- **Official Public Repository Link:** [Kaggle Telco Customer Churn (by BlastChar)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Secondary Mirror:** [IBM GitHub Telco Customer Churn ICP4D Repository](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
- **Records:** 7,043 customer accounts
- **Features:** 21 attributes encompassing customer demographics, account tenure, subscribed services, contract arrangements, payment channels, monthly charges, and churn status.

---

## 🏛️ Business Intelligence (BI) Decision Hierarchy

| Tier | Focus Question | Finding & Business Strategy |
|---|---|---|
| **Level 1: KPIs** | *What is happening?* | 7,043 total customers, **26.54% overall churn rate**, **$139,131 Monthly Recurring Revenue (MRR) at risk** out of $456,117 total MRR. |
| **Level 2: Trends** | *Where is it moving?* | Severe **tenure hazard curve**: 47.4% churn in months 0–12 vs < 9.8% churn for customers with tenure > 48 months. |
| **Level 3: Drivers** | *Why is it happening?* | **Month-to-month contracts** (42.7% churn), **Fiber Optic service lacking Tech Support** (49.3% churn), and **Electronic Check payment friction** (45.3% churn). |
| **Level 4: Risks & Opportunities** | *What could go wrong or improve?* | Month-to-month subscribers with charges >$70/mo account for 68% of lost revenue. Transitioning 15% to 1-year contracts recovers **~$252,000 ARR**. |
| **Level 5: Actions** | *What should management do?* | Deploy automated retention playbooks: annual contract migration discounts (15%), 90-day complimentary tech support bundles, and ACH payment incentives. |

---

## 🤖 Machine Learning Model Benchmarks

Three supervised machine learning classifiers were trained and rigorously evaluated using an 80/20 stratified train/test split:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression (Baseline)** | 80.70% | 65.84% | 56.68% | 60.92% | 0.8416 |
| **Random Forest Classifier (150 trees)** | 80.48% | 67.49% | 51.07% | 58.14% | 0.8442 |
| **Gradient Boosting Classifier (Selected Best)** | **80.62%** | **67.00%** | **53.21%** | **59.31%** | **0.8426** |

**Selected Production Model:** **Gradient Boosting Classifier** was selected for interactive inference due to its calibrated probability distribution and optimal balance of precision and recall.

---

## 🛠️ Technology Stack
- **Programming Language:** Python 3.10+
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Machine Learning & Pipeline:** Scikit-Learn
- **Data Visualizations:** Matplotlib, Seaborn
- **Interactive Web Dashboard:** Streamlit
- **Academic Documentation:** Python-Docx (Word `.docx`)

---

## 💻 Installation & Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Sanketh1731/AICTE-IBM-Customer-Churn-Prediction.git
cd AICTE-IBM-Customer-Churn-Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Model Training & Evaluation Pipeline (CLI)
```bash
python Sanketh_CustomerChurnPrediction.py
```
*This command runs data preprocessing, trains all models, prints the benchmark table to the console, and exports high-resolution visual assets to the `assets/` directory.*

### 4. Launch the Interactive BI Web Dashboard
```bash
streamlit run Sanketh_CustomerChurnPrediction.py
```
Open your web browser and navigate to `http://localhost:8501` to view:
- **Page 1: Executive Overview** (KPI cards, tenure hazard curve, contract breakdown).
- **Page 2: Drivers & Deep Dive** (Payment channels, tech support impact, ML feature importances).
- **Page 3: AI Churn Predictor & Simulator** (Real-time customer risk simulation with automated retention recommendations).

---

## 📋 Submission Checklist & Required Files

This repository contains all mandatory files required for the **AICTE | IBM SkillsBuild Internship** final submission:

| Deliverable | File Name | Description | Status |
|---|---|---|---|
| **1. Code File** | `Sanketh_CustomerChurnPrediction.py` | Complete end-to-end Python file (Data pipeline + ML models + Streamlit dashboard) | ✅ Complete |
| **2. Requirements File** | `requirements.txt` | Complete list of Python libraries and dependencies | ✅ Complete |
| **3. Project Report** | `Sanketh_ProjectReport.docx` | Comprehensive documentation in Microsoft Word format with embedded figures | ✅ Complete |
| **4. README File** | `README.md` | Full repository documentation with dataset link and instructions | ✅ Complete |
| **5. GitHub Repository Link** | `https://github.com/Sanketh1731/AICTE-IBM-Customer-Churn-Prediction` | Public GitHub link submitted in the Google Form | ✅ Ready |

---

## 👤 Author Information
- **Name:** Sanketh
- **Email:** sankethbkr2005@gmail.com
- **Internship Program:** AICTE | IBM SkillsBuild Data Analytics with AI Academic Internship 2026
- **Training Partner:** BharatCares (SMEC Trust)
