import pandas as pd
import numpy as np
import lightgbm as lgb
import time
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

def main():
    print("=== STARTING LIGHTGBM BENCHMARK ===")
    
    # 1. Load Data
    print("Loading dataset 'creditcard.csv'...")
    start_load = time.time()
    try:
        df = pd.read_csv('creditcard.csv')
    except FileNotFoundError:
        print("Error: creditcard.csv not found in the current directory.")
        print("Please ensure you have run the kaggle download and unzip commands first.")
        return
    load_time = time.time() - start_load
    print(f"Data loaded successfully in {load_time:.4f} seconds.")
    print(f"Dataset shape: {df.shape}")

    # 2. Split Data
    X = df.drop(columns=['Class'])
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train LightGBM Model
    print("Training LightGBM model (100 estimators)...")
    # Setting n_jobs=-1 to utilize all 8 vCPUs of e2-standard-8
    model = lgb.LGBMClassifier(n_estimators=100, n_jobs=-1, random_state=42, verbose=-1)
    
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train
    print(f"Training completed in {train_time:.4f} seconds.")

    # 4. Evaluate Metrics
    print("Evaluating metrics on test set...")
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_pred_prob)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    # 5. Measure Inference Latency (1 row)
    print("Measuring inference latency (average of 100 runs for a single row)...")
    single_row = X_test.iloc[[0]]
    latencies = []
    for _ in range(100):
        t_start = time.time()
        model.predict(single_row)
        latencies.append(time.time() - t_start)
    avg_latency = np.mean(latencies) * 1000  # convert to ms

    # 6. Measure Throughput (1000 rows)
    print("Measuring inference throughput for 1000 rows...")
    thousand_rows = X_test.iloc[:1000]
    t_start = time.time()
    model.predict(thousand_rows)
    tp_time = time.time() - t_start
    throughput = 1000 / tp_time  # rows per second

    # Compile results
    results = {
        "data_loading_time_seconds": round(load_time, 4),
        "training_time_seconds": round(train_time, 4),
        "auc_roc": round(auc, 6),
        "accuracy": round(acc, 6),
        "f1_score": round(f1, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "inference_latency_1row_ms": round(avg_latency, 4),
        "inference_throughput_1000rows_per_second": round(throughput, 2)
    }

    # Save to file
    with open('benchmark_result.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Results saved to 'benchmark_result.json'.")

    # Display results
    print("\n" + "="*40)
    print("            BENCHMARK RESULTS")
    print("="*40)
    print(f"Data Loading Time:  {results['data_loading_time_seconds']} s")
    print(f"Training Time:      {results['training_time_seconds']} s")
    print(f"AUC-ROC Score:      {results['auc_roc']:.6f}")
    print(f"Accuracy:           {results['accuracy']:.6f}")
    print(f"F1-Score:           {results['f1_score']:.6f}")
    print(f"Precision:          {results['precision']:.6f}")
    print(f"Recall:             {results['recall']:.6f}")
    print(f"Latency (1 row):    {results['inference_latency_1row_ms']:.4f} ms")
    print(f"Throughput (1000):  {results['inference_throughput_1000rows_per_second']:.2f} rows/s")
    print("="*40)

if __name__ == '__main__':
    main()
