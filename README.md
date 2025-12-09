# ReAct-Based Hybrid Fraud Detection Framework
Hybrid AI System for Real-Time Healthcare Insurance Fraud Detection

## Overview
This project implements a ReAct-based hybrid fraud detection framework that integrates machine learning–based anomaly detection with large language model (LLM) reasoning. The system is designed for real-time detection of fraudulent healthcare insurance claims, manipulated medical reports, and billing irregularities.

The approach combines a Random Forest classifier with an LLM (Gemini 2.5 Flash Lite) using a structured Reason–Act–Verify loop. The LLM interprets and validates the ML predictions, generating clear and interpretable explanations for each decision.

## Key Features
1. Hybrid AI Fraud Detection
   - Random Forest performs anomaly detection on structured claim data.
   - LLM provides reasoning-based validation and explanation.

2. ReAct Loop (Reason → Act → Verify)
   - Reason: LLM analyzes claim details and contextual indicators.
   - Act: ML model produces fraud probability and classification.
   - Verify: LLM checks consistency of outputs and provides justification.

3. Explainable and Interpretable Outputs
   - Each prediction is accompanied by a human-readable explanation.
   - Useful for insurance audits and medical claim reviews.

4. Cloud-Ready and Scalable
   - Lightweight Random Forest model.
   - Efficient LLM (Gemini 2.5 Flash Lite) suitable for real-time cloud deployment.

## Methodology
1. Preprocessed a synthetic dataset of 36,000 healthcare claim records.
2. Trained a Random Forest classifier for fraud/anomaly detection.
3. Developed a ReAct-based loop integrating ML and LLM reasoning.
4. Performed consistency checks between model outputs and LLM interpretations.
5. Evaluated performance against zero-shot LLM-only baselines.

## Results
- Accuracy: 83%
- F1-Score: 0.80
- Recall: 51% higher than zero-shot LLM baselines
- Demonstrated that hybrid ML + LLM reasoning enhances accuracy, reliability, and interpretability.

## Tech Stack
- Python
- Scikit-learn
- Gemini 2.5 Flash Lite
- Pandas, NumPy
- ReAct Agent Methodology

## How to Run
1. Install dependencies:
   pip install -r requirements.txt

2. Train the Random Forest model:
   python src/train_model.py

3. Run the hybrid ReAct fraud detection pipeline:
   python src/react_pipeline.py

4. Input claim data to receive:
   - ML prediction
   - LLM reasoning
   - Final hybrid fraud decision
   - Explanation text

## Conclusion
This project demonstrates the effectiveness of combining machine learning with LLM reasoning for real-time healthcare fraud detection. The ReAct-based hybrid model significantly improves accuracy, interpretability, and trustworthiness, making it suitable for practical deployment in cloud-based healthcare and insurance environments.
