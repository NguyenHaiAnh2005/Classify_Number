# 🔢 Classify_Number
> Nhận dạng chữ số viết tay trên tập dữ liệu MNIST.

## 📖 Giới thiệu

**Classify_Number** là đồ án môn **Nhận dạng mẫu**, tập trung xây dựng
và đánh giá các mô hình phân loại chữ số viết tay từ **0 đến 9**.

Dự án sử dụng tập dữ liệu **MNIST**, trong đó mỗi mẫu là một ảnh xám
có kích thước **28 × 28 pixel**, tương ứng với **784 đặc trưng pixel**.

## 🎯 Mục tiêu

- Khám phá dữ liệu và xây dựng quy trình tiền xử lý.
- Triển khai, huấn luyện và so sánh các mô hình nhận dạng.
- Phân tích kết quả, ưu điểm và hạn chế của từng phương pháp.
- Xây dựng giao diện **Streamlit** để người dùng vẽ chữ số và xem dự đoán.

## 🧠 Các mô hình dự kiến

| Mô hình | Hướng tiếp cận |
|---|---|
| Gaussian Naive Bayes | Phân loại dựa trên xác suất |
| Logistic Regression | Phân loại tuyến tính đa lớp |
| Random Forest | Kết hợp nhiều cây quyết định |
| Multilayer Perceptron — MLP | Mạng neural truyền thẳng |
| Convolutional Neural Network — CNN | Mạng neural tích chập cho ảnh |

## 📊 Quy trình đánh giá

Dữ liệu được chia thành ba tập với các vai trò riêng:

- **Train:** huấn luyện mô hình.
- **Validation:** lựa chọn cấu hình và theo dõi quá trình huấn luyện.
- **Test:** đánh giá cuối sau khi chốt phương án.

Các tiêu chí so sánh gồm **Accuracy**, **Macro F1-score**,
**Confusion Matrix**, thời gian huấn luyện và thời gian dự đoán.

## 🛠️ Công nghệ sử dụng

- **Python:** ngôn ngữ lập trình chính.
- **NumPy, Pandas:** xử lý dữ liệu.
- **Matplotlib, Seaborn:** trực quan hóa.
- **Scikit-learn, PyTorch:** hỗ trợ xây dựng và đánh giá mô hình.
- **Streamlit:** xây dựng giao diện tương tác.
- **Jupyter Notebook:** khám phá dữ liệu và thực nghiệm.

---

*Đồ án được thực hiện bởi nhóm 3 thành viên, hướng đến vận dụng
kiến thức nhận dạng mẫu vào bài toán phân loại chữ số viết tay.*