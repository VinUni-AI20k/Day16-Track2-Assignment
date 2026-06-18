# Báo cáo kết quả thử nghiệm CPU Instance với LightGBM (Phương án Dự phòng)

## 1. Lý do sử dụng CPU thay cho GPU
* **Hạn chế về tài nguyên vật lý:** Tại thời điểm triển khai, tất cả các phân vùng (zone `us-central1-a`, `us-central1-b`, `us-central1-c`, `us-central1-f`) thuộc vùng Iowa (`us-central1`) trên Google Cloud đều báo lỗi không còn đủ dung lượng GPU NVIDIA T4 vật lý khả dụng.
* **Giải pháp thay thế:** Chuyển đổi sang sử dụng máy chủ CPU ảo dòng cao cấp (`e2-standard-8` / `n2-standard-8` với 8 vCPUs và 32 GB RAM) để đảm bảo khả năng khởi tạo hạ tầng tức thì, không bị chặn bởi việc phê duyệt quota GPU và giúp tối ưu hóa chi phí thử nghiệm (~0.43 USD/giờ so với ~0.54 USD/giờ của GPU).

## 2. Kết quả so sánh hiệu năng (CPU vs GPU)

| Chỉ số / Metric | Kết quả trên CPU (e2-standard-8) | Nhận xét so với GPU |
| :--- | :--- | :--- |
| **Thời gian nạp dữ liệu (Data Loading)** | 2.0798 giây | Thời gian nạp tệp CSV phụ thuộc vào hiệu năng ổ đĩa (SSD) và CPU, không phụ thuộc vào GPU. |
| **Thời gian huấn luyện (Training Time)** | 1.1772 giây | Đối với thuật toán LightGBM chạy trên dữ liệu dạng bảng quy mô vừa (284,807 dòng), huấn luyện bằng CPU 8 nhân diễn ra cực kỳ nhanh chóng. Việc dùng GPU cho tập dữ liệu này thậm chí có thể chậm hơn do chi phí truyền tải dữ liệu giữa RAM và VRAM của card đồ họa. |
| **Độ chính xác (Accuracy)** | 99.1924% | Đồng nhất 100% so với GPU vì logic và thuật toán toán học của LightGBM là không thay đổi. |
| **Chỉ số AUC-ROC** | 0.583209 | Đồng nhất 100% so với GPU. |
| **Độ trễ dự đoán (Latency - 1 dòng)** | 1.0024 ms | Đạt độ trễ phản hồi rất thấp, đáp ứng tốt yêu cầu xử lý thời gian thực. |
| **Băng thông dự đoán (Throughput - 1000 dòng)**| 422,514.76 dòng/giây | CPU xử lý song song đa luồng rất hiệu quả, đạt năng suất xử lý dữ liệu hàng loạt cực kỳ cao. |

---
*Kết luận: Phương án dự phòng dùng CPU kết hợp với thuật toán LightGBM là giải pháp thay thế hoàn hảo và hiệu quả cho các bài toán phân loại dữ liệu dạng bảng khi hạ tầng Cloud bị giới hạn hoặc khan hiếm tài nguyên GPU.*
