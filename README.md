# main.py

## Config 
The script predicts **HTTP response time (ms)** for web requests, based on network-level timing features (DNS, TCP, TLS, RTT, response size, time of day, domain category).

Đi qua 7 bước: Load → Clean → Analyze → Feature Engineer → Train → Evaluate → Visualize.

| Constant | Value | Mục đích |
|---|---|---|
| DATA_PATH | Dataset_finale.csv | Dataset đầu vào |
| OUT_DIR | outputs | Kết quả đầu ra, các file .csv, plots đều ở folder này |
| DISPLAY_NAME | dict | Bảng map các biến thành tên dễ đọc hơn. VD: "dns_time_ms": "DNS Time (ms)" |

Nếu tên file input khác phải vào code chỉnh

## 1. Load (Đọc dữ liệu)

**`load_data(path=DATA_PATH)`**
- **Input:** đường dẫn tới file CSV
- **Việc làm:** đọc CSV vào DataFrame, xóa khoảng trắng thừa ở tên cột
- **Output:** DataFrame thô (chưa xử lý)
- **In ra:** shape của dữ liệu vừa load

---

## 2. Clean (Làm sạch dữ liệu)

### `clean_structural(df)`
Làm sạch cấu trúc, thực hiện **trước** khi chia train/test (an toàn vì không "học" gì từ dữ liệu, chỉ loại bỏ các dòng rõ ràng không hợp lệ).

- **Input:** DataFrame thô
- **Việc làm:**
  1. Xóa các dòng có giá trị null và các dòng trùng lặp
  2. Chỉ giữ lại các dòng có `http_status == 200` (request thành công), sau đó xóa cột `http_status`
  3. Xóa các dòng mà tổng `dns_time_ms + tcp_connect_ms + tls_time_ms` lớn hơn `response_time_ms` (dữ liệu bất khả thi/lỗi)
  4. Chuẩn hóa text của `domain_category` (xóa khoảng trắng, chuyển về chữ thường)
- **Output:** DataFrame đã làm sạch
- **In ra:** số lượng null/duplicate và số dòng bị xóa ở mỗi bước

### `split_data(df, test_size=0.2, random_state=42)`
- **Input:** DataFrame đã làm sạch
- **Việc làm:** chia thành tập train/test (mặc định 80/20) dùng `train_test_split`
- **Output:** `(train_df, test_df)`
- **In ra:** shape của cả 2 tập

### `clean_statistical(train_df, test_df, numeric_cols)`
Loại bỏ outlier bằng mô hình thống kê — thực hiện **sau** khi chia tập, để mô hình phát hiện outlier chỉ "học" từ tập train, tránh rò rỉ dữ liệu (data leakage).

- **Input:** `train_df`, `test_df`, danh sách các cột số cần kiểm tra
- **Việc làm:** fit `IsolationForest` (contamination 2%) trên các cột số của tập train, sau đó áp dụng cùng mô hình đã fit để đánh dấu/loại bỏ outlier ở cả train và test
- **Output:** `(train_clean, test_clean)` — DataFrame đã loại bỏ các dòng bất thường
- **In ra:** số lượng outlier bị xóa ở mỗi tập

---

## 3. Analyze (Phân tích)

**`analyze(df, label="train")`**
- **Input:** một DataFrame và một chuỗi label để log
- **Việc làm:** in ra thống kê tổng quan (`.describe()`) và value counts của `domain_category`
- **Output:** không có (chỉ in ra console — bước khám phá dữ liệu)

---

## 4. Feature Engineering (Kỹ thuật đặc trưng)

### `encode_time_cyclical(df)`
- **Input:** DataFrame có cột `time_of_day` dạng số nguyên `HHMMSS` (VD: `143000` = 14:30:00)
- **Việc làm:** parse giờ/phút/giây, kiểm tra tính hợp lệ, chuyển thời gian trong ngày thành góc trên vòng tròn 24 giờ, mã hóa thành 2 đặc trưng liên tục `time_sin` và `time_cos` (để 23:59 và 00:01 được nhận diện là "gần nhau" về mặt thời gian). Xóa cột `time_of_day` gốc.
- **Output:** DataFrame với `time_sin`, `time_cos` thay thế `time_of_day`
- **Raises:** `ValueError` nếu có giá trị thời gian không hợp lệ (giờ > 23, phút/giây > 59)

### `feature_engineer(train_df, test_df)`
- **Input:** DataFrame train và test (sau khi đã làm sạch thống kê)
- **Việc làm:**
  1. Áp dụng `encode_time_cyclical` cho cả 2
  2. One-hot encode `domain_category` (bỏ category đầu tiên để tránh đa cộng tuyến), tập category chỉ được fit trên **train**
  3. Reindex cột của test để khớp với cột của train (điền 0 vào các cột dummy bị thiếu), đảm bảo train/test có cùng tập đặc trưng dù có category nào đó thiếu ở một bên
- **Output:** `(train_df, test_df)` với các đặc trưng đã được kỹ thuật hóa/mã hóa
- **In ra:** danh sách cột cuối cùng của tập train

### `scale_features(train_df, test_df, feature_cols)`
- **Input:** DataFrame train/test và danh sách cột đặc trưng cần scale
- **Việc làm:** fit `StandardScaler` chỉ trên đặc trưng của train, sau đó transform cả train và test bằng cùng scaler đó (một lần nữa để tránh rò rỉ dữ liệu)
- **Output:** `(train_scaled, test_scaled, scaler)`

---

## 5. Model Training (Huấn luyện mô hình)

**`train_models(X_train, y_train)`**
- **Input:** đặc trưng train (`X_train`) và target (`y_train` = `response_time_ms`)
- **Việc làm:** huấn luyện 4 mô hình:
  1. `baseline_mean` — baseline đơn giản, luôn dự đoán giá trị trung bình của `y_train` (class tự viết `MeanBaseline`)
  2. `linear_regression` — `LinearRegression`
  3. `decision_tree` — `DecisionTreeRegressor` (max depth 6)
  4. `random_forest` — `RandomForestRegressor` (200 cây, max depth 8)
- **Output:** `dict` map tên mô hình → object mô hình đã fit
- **In ra:** danh sách tên các mô hình đã huấn luyện

---

## 6. Evaluation (Đánh giá)

**`evaluate(models, X_test, y_test)`**
- **Input:** dict các mô hình đã huấn luyện, đặc trưng test, target test
- **Việc làm:** với mỗi mô hình, sinh dự đoán trên tập test và tính:
  - **MAE** (Mean Absolute Error)
  - **RMSE** (Root Mean Squared Error)
  - **R²** (hệ số xác định)
  - **MAPE %** (Mean Absolute Percentage Error)
  - **Accuracy %** (`100 − MAPE`)
- **Output:** `(result, preds)`
  - `result`: DataFrame chứa metric của từng mô hình (mỗi mô hình 1 dòng)
  - `preds`: dict map tên mô hình → mảng dự đoán
- **In ra:** bảng metric

---

## 7. Visualization (Trực quan hóa)

### `visualize_feature_target(df, target="response_time_ms")`
Bước khám phá dữ liệu, độc lập với bất kỳ mô hình nào — thể hiện mối quan hệ thô giữa từng đặc trưng số và target.

- **Input:** một DataFrame (thường là `train_df` sau khi làm sạch thống kê, trước khi scale) và tên cột target
- **Việc làm:** với mỗi cột số (trừ target), tạo scatter plot giữa đặc trưng đó và target
- **Output:** file PNG lưu tại `outputs/feature_vs_target/<feature>_vs_target.png`

### `visualize(y_test, preds, results, model_names=None)`
Các biểu đồ chẩn đoán cho từng mô hình.

- **Input:** giá trị thật của test (`y_test`), dict dự đoán (`preds`), DataFrame metric (`results`), danh sách tên mô hình cần vẽ (tùy chọn, mặc định là tất cả)
- **Việc làm:** với mỗi mô hình, tạo 3 biểu đồ và lưu vào `outputs/<model_name>/`:
  1. **`actual_vs_predicted.png`** — scatter giữa giá trị thật và giá trị dự đoán, kèm đường tham chiếu "dự đoán hoàn hảo" và ô metric (MAE, RMSE, R², Accuracy)
  2. **`residual_plot.png`** — residual (thật − dự đoán) theo giá trị dự đoán, kèm đường tham chiếu sai số bằng 0
  3. **`residual_hist.png`** — histogram của residual
- **Output:** file PNG trên ổ đĩa (hàm này không trả về giá trị)

---

## Thứ tự chạy Pipeline (`if __name__ == "__main__":`)
load_data

  → clean_structural
    → split_data                     (train/test split happens here)
      → clean_statistical            (fit on train, apply to both)
        → analyze(train)
        → visualize_feature_target(train)
          → feature_engineer         (encode time, one-hot domain_category)
            → scale_features         (fit scaler on train, apply to both)
              → train_models         (baseline, linear reg, decision tree, random forest)
                → evaluate           (MAE/RMSE/R²/MAPE/Accuracy per model)
                  → visualize        (per-model plots)
                    → save model_results.csv
