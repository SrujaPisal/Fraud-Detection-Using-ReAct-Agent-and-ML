import pandas as pd
import google.generativeai as genai
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import time
import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.calibration import CalibratedClassifierCV

# api key
genai.configure(api_key="AIzaSyBHeX2ixjQ-yolSvLM-Htb7nOOKbrmKf6c") 

# Load Datasets
train_df = pd.read_csv("/Users/srujapisal/Desktop/RESEARCHPAPERFRAUD/medical_insurance_training_dataset_30000.csv")
test_df = pd.read_csv("/Users/srujapisal/Desktop/RESEARCHPAPERFRAUD/medical_insurance_testing_dataset_6000.csv")

print(f"Train: {train_df.shape}, Test: {test_df.shape}")

eval_path = input("\nEnter the path to your evaluation CSV file: ").strip()
eval_df = pd.read_csv(eval_path)
print(f"Loaded evaluation dataset: {eval_df.shape}")

# time conversion 
def convert_time_features(df):
    df['hour'] = pd.to_datetime(df['time'], errors='coerce').dt.hour
    df['minute'] = pd.to_datetime(df['time'], errors='coerce').dt.minute
    df.drop(columns=['time'], inplace=True)
    return df

# feature engineering
def add_fraud_features(df):
    df = df.copy()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    df["log_amount"] = np.log1p(df["amount"]) 
    df["is_night"] = df["hour"].apply(lambda x: 1 if x >= 22 or x < 6 else 0)
    df["is_weekend"] = (
        pd.to_datetime(df.get("date", pd.Timestamp.now()), errors="coerce")
        .dt.dayofweek.isin([5, 6])
        .astype(int)
        if "date" in df.columns else 0
    )
    df["amount_bucket"] = pd.cut(
        df["amount"],
        bins=[-1, 5000, 10000, 50000, 100000, 500000, np.inf],
        labels=[1, 2, 3, 4, 5, 6]
    )
    df["amount_bucket"] = df["amount_bucket"].cat.add_categories([0]).fillna(0).astype(int)
    provider_counts = df["provider"].value_counts()
    df["provider_claim_count"] = df["provider"].map(provider_counts).fillna(0)
    diag_avg = df.groupby("diagnosis_code")["amount"].transform("mean")
    df["amount_vs_diag_ratio"] = df["amount"] / (diag_avg + 1)
    
    if "patient_id" in df.columns and "hour" in df.columns:
        df["patient_claims_today"] = df.groupby(["patient_id", "hour"])["hour"].transform("count")
    else:
        df["patient_claims_today"] = 1
    return df

train_df = convert_time_features(train_df)
test_df = convert_time_features(test_df)
eval_df = convert_time_features(eval_df)

train_df = add_fraud_features(train_df)
test_df = add_fraud_features(test_df)
eval_df = add_fraud_features(eval_df)

# PREPROCESSING 
categorical_cols = ["action", "provider", "diagnosis_code", "diagnosis_name"]

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(train_df[categorical_cols])

def preprocess(df):
    for col in [
        "log_amount", "is_night", "amount_bucket",
        "provider_claim_count", "amount_vs_diag_ratio", "patient_claims_today"
    ]:
        if col not in df.columns:
            df[col] = 0
    if df["amount_bucket"].dtype.name == "category":
        df["amount_bucket"] = df["amount_bucket"].cat.add_categories([0]).fillna(0).astype(int)
    else:
        df["amount_bucket"] = df["amount_bucket"].fillna(0).astype(int)

    cat_encoded = encoder.transform(df[categorical_cols])
    cat_encoded_df = pd.DataFrame(
        cat_encoded,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df.index
    )

    df_num = df[[
        "amount", "hour", "minute",
        "log_amount", "is_night", "amount_bucket",
        "provider_claim_count", "amount_vs_diag_ratio",
        "patient_claims_today"
    ]].reset_index(drop=True)

    X = pd.concat([df_num, cat_encoded_df.reset_index(drop=True)], axis=1)
    y = df["label"].astype(str).str.strip().str.lower()

    return X, y

X_train, y_train = preprocess(train_df)
X_test, y_test = preprocess(test_df)

# TRAIN RANDOM FOREST MODEL
base_rf = RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1)
rf_model = CalibratedClassifierCV(base_rf, method='isotonic', cv=3)
rf_model.fit(X_train, y_train)
ml_preds = rf_model.predict(X_test)
print(" ML model trained successfully.")

# # -------------------------------------------------------
# # FEATURE IMPORTANCE ANALYSIS (ADDED)
# # -------------------------------------------------------

# # Extract RandomForest inside CalibratedClassifierCV
# rf_underlying = rf_model.base_estimator

# # Compute importances
# importances = rf_underlying.feature_importances_
# feature_names = X_train.columns

# importance_df = pd.DataFrame({
#     "feature": feature_names,
#     "importance": importances
# }).sort_values(by="importance", ascending=False)

# print("\nTop Feature Importances:")
# print(importance_df.head(20))

# # Plot top 30
# plt.figure(figsize=(10, 12))
# plt.barh(importance_df["feature"][:30], importance_df["importance"][:30])
# plt.gca().invert_yaxis()
# plt.xlabel("Importance Score")
# plt.title("Top 30 Feature Importances - RandomForest")
# plt.tight_layout()
# plt.show()

# # -------------------------------------------------------
# # END OF FEATURE IMPORTANCE SECTION
# # -------------------------------------------------------
# -------------------------------------------------------
# FEATURE IMPORTANCE ANALYSIS (FIXED)
# -------------------------------------------------------

# Extract underlying RandomForest (first calibrated fold)
rf_underlying = rf_model.calibrated_classifiers_[0].estimator

# Compute importances
importances = rf_underlying.feature_importances_
feature_names = X_train.columns

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nTop Feature Importances:")
print(importance_df.head(20))

# Plot top 30
plt.figure(figsize=(10, 12))
plt.barh(importance_df["feature"][:30], importance_df["importance"][:30])
plt.gca().invert_yaxis()
plt.xlabel("Importance Score")
plt.title("Top 30 Feature Importances - RandomForest")
plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)


# Prepare eval data
X_eval, y_eval = preprocess(eval_df)
eval_probs = rf_model.predict_proba(X_eval)
eval_preds = rf_model.predict(X_eval)
eval_conf = np.max(eval_probs, axis=1)
eval_df["ml_prediction"] = eval_preds
eval_df["ml_confidence"] = eval_conf

# FORMAT LOGS
def format_log(row):
    return (
        f"patient_id {row['patient_id']}, action '{row['action']}', "
        f"time {row['hour']:02d}:{row['minute']:02d}, amount {row['amount']}, "
        f"provider '{row['provider']}', diagnosis_code '{row['diagnosis_code']}', "
        f"diagnosis_name '{row['diagnosis_name']}'"
    )

# initialize LLMs
llm_model = genai.GenerativeModel("gemini-2.5-flash-lite")
react_model = genai.GenerativeModel("gemini-2.5-flash-lite")

BATCH_SIZE_LLM = 30
SLEEP_TIME_LLM = 8

llm_prompt_template = """
You are a fraud detection AI.
Analyze each health insurance claim and classify it as either 'fraud' or 'non-fraud'.
Output only one word: fraud or non-fraud.

Log:
{log_text}
"""

def clean_label(text):
    text = text.strip().lower()
    if "fraud" in text and "non" not in text:
        return "fraud"
    elif "non" in text:
        return "non-fraud"
    else:
        return random.choice(["fraud", "non-fraud"])

def llm_only_classify_fast(row):
    log_text = format_log(row)
    prompt = llm_prompt_template.format(log_text=log_text)
    try:
        response = llm_model.generate_content(prompt)
        return clean_label(response.text)
    except Exception:
        return random.choice(["fraud", "non-fraud"])

print("\n Generating Zero-Shot LLM predictions...")
llm_only_preds = []

for batch_idx in tqdm(range(0, len(eval_df), BATCH_SIZE_LLM), desc="LLM-only Batches"):
    batch = eval_df.iloc[batch_idx:batch_idx + BATCH_SIZE_LLM]
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(llm_only_classify_fast, [row for _, row in batch.iterrows()]))
    llm_only_preds.extend(results)
    time.sleep(SLEEP_TIME_LLM)

# HYBRID REACT + RF
react_prompt_insurance_hybrid = """
You are an expert healthcare insurance fraud investigator AI assisting a fraud detection system.

( prompt text unchanged ... )
"""

def react_hybrid_classify_fast(row):
    log_text = format_log(row)
    ml_pred = row["ml_prediction"]
    ml_conf = row["ml_confidence"]
    
    prompt = react_prompt_insurance_hybrid.format(
        log_text=log_text, 
        ml_pred=ml_pred, 
        ml_conf=ml_conf
    )
    
    try:
        response = react_model.generate_content(prompt, generation_config={"temperature": 0.2})
        return clean_label(response.text)
    except Exception:
        return ml_pred.lower()

BATCH_SIZE = 30
SLEEP_TIME = 8
react_hybrid_preds = []

print("\n Generating Hybrid ReAct + RF predictions...")

for batch_idx in tqdm(range(0, len(eval_df), BATCH_SIZE), desc="ReAct Batches"):
    batch = eval_df.iloc[batch_idx:batch_idx + BATCH_SIZE]
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(react_hybrid_classify_fast, [row for _, row in batch.iterrows()]))
    react_hybrid_preds.extend(results)
    time.sleep(SLEEP_TIME)

# EVALUATION
def evaluate(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    print(f"\n{name} Results:")
    print(f"Accuracy:  {acc:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1-score:  {f1:.2f}")

evaluate(y_eval, llm_only_preds, "Zero-Shot LLM Only")
evaluate(y_eval, react_hybrid_preds, "Hybrid ReAct + RF")

# VISUALIZATION
def evaluate_with_return(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return acc, precision, recall, f1

llm_acc, llm_prec, llm_rec, llm_f1 = evaluate_with_return(y_eval, llm_only_preds)
react_acc, react_prec, react_rec, react_f1 = evaluate_with_return(y_eval, react_hybrid_preds)

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
llm_scores = [llm_acc, llm_prec, llm_rec, llm_f1]
react_scores = [react_acc, react_prec, react_rec, react_f1]

plt.figure(figsize=(8, 6))
x = range(len(metrics))
bar_width = 0.35
plt.bar([i - bar_width/2 for i in x], llm_scores, width=bar_width, label='LLM Only', alpha=0.7, color='salmon')
plt.bar([i + bar_width/2 for i in x], react_scores, width=bar_width, label='Hybrid ReAct + RF', alpha=0.7, color='skyblue')
plt.xticks(x, metrics, fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.ylim(0, 1)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("model_comparison_chart.png", dpi=300)
plt.show()
