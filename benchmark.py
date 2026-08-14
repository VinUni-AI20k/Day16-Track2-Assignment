import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


DATA_PATH = Path.home() / "ml-benchmark" / "creditcard.csv"
RESULT_PATH = Path.home() / "ml-benchmark" / "benchmark_result.json"

start = time.perf_counter()
df = pd.read_csv(DATA_PATH)
load_time = time.perf_counter() - start

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",
    random_state=42,
    n_jobs=2,
)

start = time.perf_counter()
model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="auc",
    callbacks=[early_stopping(30, verbose=False)],
)
training_time = time.perf_counter() - start

start = time.perf_counter()
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.5).astype(int)
inference_time = time.perf_counter() - start

one_row = X_test.iloc[[0]]
for _ in range(10):
    model.predict_proba(one_row)

start = time.perf_counter()
for _ in range(100):
    model.predict_proba(one_row)
latency_ms = (time.perf_counter() - start) / 100 * 1000

batch = X_test.iloc[:1000]
for _ in range(3):
    model.predict_proba(batch)

start = time.perf_counter()
for _ in range(10):
    model.predict_proba(batch)
throughput_rows_per_sec = 10000 / (time.perf_counter() - start)

result = {
    "dataset_rows": int(len(df)),
    "dataset_columns": int(len(X.columns)),
    "load_time_sec": load_time,
    "training_time_sec": training_time,
    "best_iteration": int(model.best_iteration_),
    "auc_roc": roc_auc_score(y_test, probabilities),
    "accuracy": accuracy_score(y_test, predictions),
    "f1_score": f1_score(y_test, predictions, zero_division=0),
    "precision": precision_score(y_test, predictions, zero_division=0),
    "recall": recall_score(y_test, predictions, zero_division=0),
    "inference_time_sec": inference_time,
    "inference_latency_ms": latency_ms,
    "inference_throughput_rows_per_sec": throughput_rows_per_sec,
}

RESULT_PATH.write_text(json.dumps(result, indent=2))

print(json.dumps(result, indent=2))
print(f"\nSaved result to: {RESULT_PATH}")