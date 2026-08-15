# Báo cáo ngắn: Benchmark LightGBM trên CPU (t3.small)

Huấn luyện LightGBM trên bộ Credit Card Fraud Detection (284,807 giao dịch, 227,845 dòng train / 56,962 dòng test) trên compute node CPU `t3.small` (2 vCPU / 2GB RAM) chỉ mất **38.4 giây** cho 1000 cây (`best_iteration = 1000`), trong khi thời gian load dữ liệu chỉ **2.1 giây** — cho thấy với dataset cỡ vừa (~150MB), CPU thường (không GPU) vẫn xử lý training trong thời gian chấp nhận được mà không cần hạ tầng đắt tiền.

Về chất lượng mô hình, AUC-ROC đạt **0.852**, cho thấy model phân biệt được giao dịch gian lận và hợp lệ ở mức khá, dù chưa tối ưu (do dùng tham số mặc định, chưa xử lý mất cân bằng lớp bằng `scale_pos_weight`/`is_unbalance`). Accuracy rất cao (**99.72%**) nhưng không phản ánh đúng chất lượng model vì dataset cực kỳ mất cân bằng (chỉ ~0.17% giao dịch là gian lận) — Precision (**0.352**) và Recall (**0.714**) là các chỉ số đáng tin cậy hơn: model bắt được ~71% giao dịch gian lận thực sự, đổi lại có khá nhiều cảnh báo sai (F1 = **0.471**).

Về tốc độ inference, độ trễ dự đoán 1 dòng chỉ **~1.35ms**, và throughput đạt **~7,133 dòng/giây** khi dự đoán theo batch — đủ nhanh để phục vụ một API inference thời gian thực với tải vừa phải, dù chỉ chạy trên CPU 2 vCPU không có GPU.

**Kết luận:** với bài toán fraud detection quy mô này, một CPU instance nhỏ (`t3.small`) đã đủ để vừa train vừa serve inference với tốc độ hợp lý, không nhất thiết cần GPU — đúng như mục tiêu của luồng chính trong lab. Để cải thiện thêm, có thể tinh chỉnh `scale_pos_weight`, bật early stopping thực sự (giảm số cây không cần thiết), và thử ngưỡng phân loại khác 0.5 để cân bằng lại Precision/Recall.
