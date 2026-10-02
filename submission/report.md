# BÁO CÁO KẾT QUẢ LAB 16 - LUỒNG CPU (TRACK DATA SCIENCE)

**Học viên:** [Dương Hữu Đạt]
**MSSV:** [2A202602544]
**Ngày thực hiện:** 02/10/2026

---

## I. THỐNG KÊ QUAN SÁT (Theo mẫu 8 dòng)

1. Tôi dùng AWS, region us-east-1, instance type là `t3.micro` (bypass Free Tier quota).
2. Dataset Credit Card Fraud có 284.807 dòng, chia Train 60% / Validation 20% / Test 20%, seed 42.
3. Load dữ liệu mất 2.49 giây; training mất 2.66 giây; best iteration là 1.
4. AUC 0.938, Accuracy 0.999, F1 0.735, Precision 0.632, Recall 0.878 trên tập test.
5. Latency 1 dòng 1.17 ms; throughput batch 1.000 dòng 678.017 dòng/giây; cách đo lấy trung bình vòng lặp.
6. CPU/RAM/Network tôi quan sát lúc chạy xong benchmark là rất nhàn rỗi (CPU gần 0%, RAM trống nhiều); ảnh đính kèm: `top.png`, `free-h.png`, `ip-s link.png`, và `Terminal_Benchmark.png`.
7. Billing tại thời điểm làm lab ghi nhận chi phí hiện tại là $0.57; ước tính riêng chi phí phát sinh từ lab (NAT Gateway, ALB, t3.micro) là khoảng ~$0.10/giờ; ảnh đính kèm: `AWS_Billing.jpg`.
8. Tôi đã tải kết quả và xóa tài nguyên lúc 17:48 (02/10/2026); bằng chứng dọn dẹp là đã chạy `terraform destroy` thành công.

---

## II. CÂU HỎI LÝ THUYẾT (CP3)

**1. Giải thích về Split Ratio (0.25):**
Việc chia 0.25 (tức là Validation 20% và Test 20% từ phần còn lại) giúp mô hình có một tập Validation riêng biệt để tinh chỉnh siêu tham số (Hyperparameter tuning) và Early Stopping trong lúc huấn luyện. Sau đó, tập Test cuối cùng hoàn toàn "lạ lẫm" với mô hình sẽ được dùng để đánh giá khách quan nhất năng lực thực tế, tránh hiện tượng học vẹt (overfitting).

**2. Tại sao lại dùng AUC thay vì tính Accuracy trực tiếp:**
Trong bài toán phát hiện gian lận thẻ tín dụng, dữ liệu bị mất cân bằng cực kỳ nghiêm trọng (imbalanced) — số giao dịch lừa đảo chỉ chiếm vỏn vẹn gần 0.17%. Nếu mô hình cứ đoán "Không lừa đảo" cho tất cả mọi giao dịch, nó vẫn đạt Accuracy 99.83%. Do đó, chỉ số AUC-ROC được sử dụng vì nó đánh giá khả năng mô hình "xếp hạng" và phân tách xác suất giữa giao dịch thật và giả một cách công bằng hơn, bất chấp tỷ lệ chênh lệch dữ liệu.

**3. Tại sao thao tác Inference (dự đoán) lại quan trọng đối với kỹ sư MLOps?**
Với MLOps, việc huấn luyện mô hình (Training) chỉ diễn ra định kỳ (vài ngày hoặc vài tuần một lần), nhưng việc dự đoán (Inference) thì diễn ra liên tục 24/7 theo thời gian thực (Real-time). Trong hệ thống ngân hàng, thời gian để quyết định một giao dịch là hợp lệ hay lừa đảo chỉ được phép tính bằng mili-giây (Latency thấp) và hệ thống phải chịu tải hàng ngàn giao dịch cùng lúc (Throughput cao). Kỹ sư MLOps phải tối ưu hóa hạ tầng và code để đáp ứng các chỉ số ngặt nghèo này.
