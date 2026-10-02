import pandas as pd
import time
import json
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

results = {}

# 1. Load data
start = time.perf_counter()
df = pd.read_csv("creditcard.csv")
results["data_load_seconds"] = time.perf_counter() - start

# 2. Tách dữ liệu (Train 60% / Val 20% / Test 20%)
X = df.drop(columns=["Class"])
y = df["Class"]

# Lần 1: Tách 20% cho tập Test, còn lại 80% (tạm gọi là Temp)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Lần 2: Tách 25% TỪ TẬP TEMP (25% của 80% = 20%) để làm tập Validation
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

results["split"] = "Train 60% / Val 20% / Test 20%"
results["seed"] = 42

# 3. Training với Early Stopping
# Dùng tập Validation để dừng sớm nếu mô hình không tốt lên, giúp chống Overfitting
model = lgb.LGBMClassifier(n_estimators=1000, random_state=42)

start = time.perf_counter()
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
)
results["training_seconds"] = time.perf_counter() - start
results["best_iteration"] = model.best_iteration_

# 4. Đánh giá (Evaluate)
# Chỉ dùng tập Test (dữ liệu mô hình chưa từng thấy) để đánh giá cuối cùng
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

results["auc_roc"] = roc_auc_score(y_test, y_prob)
results["accuracy"] = accuracy_score(y_test, y_pred)
results["f1"] = f1_score(y_test, y_pred)
results["precision"] = precision_score(y_test, y_pred)
results["recall"] = recall_score(y_test, y_pred)

# 5. Latency (Đo tốc độ dự đoán 1 dòng)
sample_1 = X_test.iloc[[0]]
# Warm-up (chạy nháp vài lần để load model vào RAM)
for _ in range(10):
    model.predict(sample_1)

# Đo thật (Chạy 100 lần lấy trung bình cho chính xác)
start = time.perf_counter()
for _ in range(100):
    model.predict(sample_1)
results["latency_1_row_ms"] = ((time.perf_counter() - start) / 100) * 1000

# 6. Throughput (Số lượng dòng xử lý được trong 1 giây)
sample_1000 = X_test.iloc[:1000]
start = time.perf_counter()
model.predict(sample_1000)
time_taken = time.perf_counter() - start
results["throughput_1000_rows_per_second"] = 1000 / time_taken

# 7. Lưu và In kết quả
with open("benchmark_result.json", "w") as f:
    json.dump(results, f, indent=4)

print("=== HOÀN TẤT BENCHMARK ===")

