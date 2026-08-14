# Báo cáo Benchmark LightGBM trên CPU

Dataset Credit Card Fraud Detection được tải và xử lý thành công trên VM GCP.
Thời gian đọc dữ liệu là **1,72 giây**, thời gian huấn luyện LightGBM là **1,63 giây**.
Mô hình đạt **AUC-ROC 0,9517** và **Accuracy 0,9989** trên tập kiểm thử.
Các chỉ số F1-score, Precision và Recall lần lượt là **0,7273**, **0,6557** và **0,8163**.
Inference latency cho một dòng là khoảng **0,0016 giây**.
Inference throughput đạt khoảng **602.136 dòng/giây** trên CPU.
Kết quả cho thấy LightGBM có thời gian huấn luyện nhanh và tốc độ suy luận tốt cho bài toán phát hiện giao dịch gian lận.
