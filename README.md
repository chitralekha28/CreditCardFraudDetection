Hybrid Autoencoder–GNN Based Fraud Detection System

A Research Implementation & Case Study Project
Overview
This repository contains the complete implementation, documentation, and outputs for the research-based project developed as part of the Coding and Data Analysis Round.
The project is based on a selected Q2/Q3 journal research paper and includes:
A unique proposed solution
Complete Python implementation
Case study
Literature review & research gaps
Visualizations
Comparative analysis
🎯 Problem Statement

Fraudulent transactions pose a major threat to financial institutions. Traditional ML models are unable to effectively detect fraud due to:

Highly imbalanced datasets

Non-linear patterns in transaction data

Limited ability to capture relational dependencies between entities

This project proposes a Hybrid Autoencoder + Graph Neural Network (GNN) framework to improve anomaly detection accuracy and robustness.

🧩 Proposed Solution

The proposed system combines:

1️⃣ Autoencoder (AE)

To learn compressed latent features and reconstruct normal transaction patterns.

2️⃣ Graph Neural Network (GNN)

To capture relationships between accounts, devices, IP addresses, and transactions.

3️⃣ Hybrid Scoring Mechanism

A combined anomaly score = reconstruction error + GNN embedding distance.

This increases accuracy and reduces false positives.

📝 Research Questions

Can hybrid deep learning models improve fraud detection compared to traditional ML models?

Does combining Autoencoders with GNN embeddings reduce false positives?

How does the hybrid model perform on imbalanced vs balanced datasets?

Can feature learning reduce dependency on manual feature engineering?

🧪 Dataset & Preprocessing

The dataset used contains:

Transaction details

User identity features

Device/behavioral features

Major preprocessing tasks:

✔ Handling missing values
✔ Normalization & standardization
✔ Label encoding
✔ Graph construction (edges between entities)
✔ Train–test split
✔ SMOTE for handling imbalance

🏗️ Architecture

The architecture includes:

Autoencoder for representation learning

Graph Neural Network (GCN / GraphSAGE)

Dense classifier

Hybrid scoring module

📊 Results & Visualizations

Stored under /Output/Visualizations/

Includes:

Confusion matrix

Accuracy graph

Loss curves

Feature importance

Comparing baseline vs proposed model

🔬 Comparative Analysis

Models compared:

Logistic Regression

Random Forest

SVM

XGBoost

Standalone Autoencoder

Standalone GNN

Proposed Hybrid AE-GNN Model (Best Performer)

Metrics:

Accuracy

F1-score

Precision

Recall

AUC-ROC
