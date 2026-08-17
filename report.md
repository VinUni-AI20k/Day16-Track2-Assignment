# Báo cáo kết quả Benchmark LightGBM (CPU — `t3.micro`)

**Dataset:** Credit Card Fraud Detection — 284,807 giao dịch, ~0.17% là gian lận.

| Chỉ số | Kết quả | Nhận xét |
|---|---|---|
| Thời gian load data | 1.93 s | Nhanh nhờ dataset chỉ ~150MB |
| Thời gian training | 2.32 s | LightGBM tối ưu tốt cho CPU, không cần GPU |
| AUC-ROC | 0.9517 | Phân biệt tốt gian lận/hợp lệ dù dữ liệu mất cân bằng nặng |
| Precision / Recall | 0.6159 / 0.8673 | Recall cao hơn Precision → mô hình ưu tiên bắt gian lận thật, chấp nhận vài cảnh báo nhầm |
| Inference latency | 1.14 ms/dòng | Đủ nhanh cho phục vụ real-time |
| Inference throughput | ~690,000 dòng/giây | Rất cao, tận dụng tốt phần cứng CPU giá rẻ |

**Kết luận:** Với chi phí hạ tầng ước tính chỉ ~$0.10/giờ, luồng CPU + LightGBM là lựa chọn hiệu quả về chi phí cho bài toán tabular data như fraud detection — không cần đến GPU đắt đỏ vốn chỉ thực sự cần thiết cho các tác vụ LLM/ngôn ngữ tự nhiên.



terraform.zip: https://github.com/HaDuyAnh025/Track2_Day16_2A202601511/releases/tag/v1.0-terraform
