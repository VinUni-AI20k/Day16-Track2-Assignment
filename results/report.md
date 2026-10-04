# Lab 16: Báo cáo benchmark LightGBM trên GCP (CPU)

## Môi trường đo

| Mục | Giá trị |
|---|---|
| Cloud / Region / Zone | GCP / `us-central1` / `us-central1-a` |
| Instance | `e2-medium`: 2 vCPU, 4 GB RAM (3.8 GiB khả dụng), Debian 12, x86_64 |
| Hạ tầng | Terraform (`terraform-gcp/`), VM trong private subnet, SSH qua IAP + OS Login, ra internet qua Cloud NAT |
| Thời điểm đo | 2026-10-02 10:34:03 UTC |
| Phần mềm | Python 3.11.2, LightGBM 4.7.0, scikit-learn 1.9.1, pandas 3.0.6, numpy 2.4.6 |
| Thời gian `terraform apply` | ≈ 3 phút 00 giây (09:49:33 → 09:52:33 UTC, 16 resource, theo Cloud Audit Logs) |
| VM sẵn sàng | Boot lúc 09:50:24, startup script cài xong thư viện lúc 09:52:01 (≈ 1.5 phút sau khi boot) |

## Dữ liệu và cách đo

- Dataset: Kaggle Credit Card Fraud Detection, 284,807 giao dịch, trong đó 492 giao dịch gian lận (0.17%).
- Chia dữ liệu 60/20/20 có stratify (seed 16): train 170,883 / validation 56,962 / test 56,962. Tập validation chỉ dùng cho early stopping. Mọi metric đều tính trên tập test.
- Mô hình: `LGBMClassifier` (learning_rate 0.05, tối đa 300 cây, early stopping 20 vòng theo AUC trên tập validation), `n_jobs=2`.
- Latency và throughput: lấy median, không tính lần warm-up, gọi `predict_proba` trên DataFrame pandas. Latency lấy median của 50 lần, mỗi lần dự đoán 1 dòng. Throughput lấy median của 10 lần, mỗi lần dự đoán 1000 dòng.

## Kết quả

| Metric | Kết quả |
|---|---|
| Thời gian load data | 2.99 s |
| Thời gian training | 4.39 s |
| Best iteration | 68 |
| AUC-ROC | 0.9768 |
| Accuracy | 0.9995 |
| F1-Score | 0.8478 |
| Precision | 0.9070 |
| Recall | 0.7959 |
| Inference latency (1 row) | 1.44 ms |
| Inference throughput (1000 rows) | 4.11 ms / 1000 dòng ≈ 243,300 dòng/giây |

Số liệu gốc: `benchmark_result.json` (lần chạy 10:34 UTC; một lần chạy trước đó lúc 10:16 cho cùng metric mô hình, thời gian chênh dưới 10%).

## Nhận xét

1. **Training rất nhanh trên CPU nhỏ.** Mô hình train xong 171k dòng trong 4.4 s trên 2 vCPU và dừng sớm ở cây thứ 68 (giới hạn là 300). Với bài toán dạng bảng cỡ này, GPU không cần thiết.
2. **AUC-ROC 0.977.** AUC được tính từ xác suất dự đoán, nên đo khả năng mô hình xếp giao dịch gian lận có điểm cao hơn giao dịch thường, trên mọi ngưỡng. Kết quả cho thấy mô hình phân tách hai lớp tốt.
3. **Accuracy 99.95% không có nhiều ý nghĩa.** Vì 99.83% giao dịch là hợp lệ, một mô hình đoán "không gian lận" cho mọi giao dịch cũng đạt khoảng 99.8%. Chỉ số đáng xem là Precision và Recall của lớp gian lận.
4. **Ở ngưỡng 0.5**, tập test có 98 giao dịch gian lận. Mô hình bắt được 78 (Recall 0.80), bỏ sót 20, và báo nhầm 8 giao dịch hợp lệ (Precision 0.91). Nếu ưu tiên bắt gian lận, có thể hạ ngưỡng để tăng Recall, đổi lại Precision sẽ giảm. AUC không đổi khi đổi ngưỡng.
5. **Latency và throughput khác nhau.** Dự đoán 1 dòng mất khoảng 1.4 ms. Dự đoán theo lô 1000 dòng chỉ mất 4.1 ms (khoảng 243k dòng/giây). Mỗi lần gọi `predict_proba` tốn một khoản chi phí cố định (chủ yếu do xử lý pandas), nên chạy theo lô hiệu quả hơn khoảng 350 lần tính theo từng dòng.
6. **Không cộng các thời gian lại với nhau.** Thời gian tạo hạ tầng, load data, training và inference là bốn phép đo riêng. Inference latency ở đây chỉ là thời gian của `predict_proba`.
7. **Chi phí.** Chi phí ước tính khoảng $0.09/giờ (VM `e2-medium`, Cloud NAT, Load Balancer). Billing Reports lúc chụp hiển thị ₫0 do dữ liệu chi phí cập nhật trễ vài giờ; chi phí được trừ vào free trial credits. Toàn bộ tài nguyên được xóa bằng `terraform destroy` sau khi thu kết quả.
