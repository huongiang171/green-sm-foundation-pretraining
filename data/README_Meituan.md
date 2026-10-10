# BỘ DỮ LIỆU MEITUAN TRD & THIẾT KẾ CÁC BÀI TOÁN HỆ THỐNG GỢI Ý

---

## 1. THÔNG TIN BỘ DỮ LIỆU & GIẢI NGHĨA CÁC TRƯỜNG THỰC TẾ

Bộ dữ liệu **Meituan TRD (Takeout Recommendation Dataset)** là tập dữ liệu thực tế do Meituan công bố, thu thập trong **28 ngày liên tục** (từ `01/03/2021` đến `28/03/2021`) tại **11 khu thương mại lớn** (`aor_id` từ 0 đến 10) ở Bắc Kinh.
* **Định dạng file:** Bảng dữ liệu dạng văn bản phân tách bằng dấu tab (TSV / `.txt`), bảng mã UTF-8.
* **Quy mô:** 200.000 người dùng, 29.071 quán ăn (POI), 179.759 món ăn (SPU), hơn 1,29 triệu đơn hàng và phiên tương tác.

### 1.1. Bảng thông tin người dùng (`users.txt`)
* **Tổng số dòng:** 200.000 dòng (tương ứng 200.000 người dùng duy nhất).

| Tên trường (Cột) | Kiểu dữ liệu | Ý nghĩa tiếng Việt | Ví dụ thực tế |
| :--- | :--- | :--- | :--- |
| `user_id` | Chuỗi ID số | Mã định danh duy nhất của người dùng. | `"0"`, `"140760"` |
| `avg_pay_amt` | Chuỗi phân khoảng | Mức chi tiêu trung bình lịch sử của người dùng (tính bằng CNY). | `"<29"`, `"[29,36)"`, `"[36,49)"`, `"[49,65)"`, `">=65"` |
| `avg_pay_amt_weekdays` | Chuỗi phân khoảng | Mức chi tiêu trung bình của người dùng vào các ngày trong tuần (Thứ 2 – Thứ 6). | `"[36,49)"` |
| `avg_pay_amt_weekends` | Chuỗi phân khoảng | Mức chi tiêu trung bình của người dùng vào hai ngày cuối tuần (Thứ 7 – Chủ Nhật). | `">=65"` |

---

### 1.2. Bảng thông tin quán ăn (`pois.txt`)
* **Tổng số dòng:** 29.072 dòng (29.071 quán ăn POI duy nhất).

| Tên trường (Cột) | Kiểu dữ liệu | Ý nghĩa tiếng Việt | Ví dụ thực tế |
| :--- | :--- | :--- | :--- |
| `wm_poi_id` | Chuỗi ID số | Mã định danh duy nhất của quán ăn / nhà hàng. | `"2198"`, `"3193"` |
| `wm_poi_name` | Chuỗi ký tự | Tên quán ăn (đã được băm/mã hóa bảo mật). | `"302658-52727-..."` |
| `primary_first_tag_name` | Chuỗi ID số | **Danh mục ngành hàng Cấp 1 (Broad Category - 16 nhóm):** Nhóm ẩm thực lớn nhất để phân loại trên app (ví dụ: Đồ ăn mặn/Cơm, Đồ uống/Trà sữa, Tráng miệng/Bánh ngọt, Lẩu/Nướng...). | `"0"`, `"1"` |
| `primary_second_tag_name`| Chuỗi ID số | **Danh mục ngành hàng Cấp 2 (Sub-category - 58 nhóm):** Thể loại món chi tiết hơn được chia nhỏ từ Cấp 1 (ví dụ: trong *Đồ ăn mặn* chia thành Cơm văn phòng, Bún/Phở/Mì, Đồ ăn nhanh gà rán/burger...). | `"0"`, `"2"` |
| `primary_third_tag_name` | Chuỗi ID số | **Danh mục ngành hàng Cấp 3 (Micro-cuisine - 124 nhóm):** Loại món ăn đặc thù/chuyên sâu nhất của quán (ví dụ: trong *Bún/Mì* chia ra Bún bò, Mì cay, Hủ tiếu... Nếu quán không đăng ký sâu đến cấp này sẽ ghi nhận là `"未知"` - Chưa xác định). | `"0"`, `"1"`, `"未知"` |
| `poi_brand_id` | Chuỗi ID số | Mã thương hiệu / chuỗi nhà hàng (0 nếu là quán nhỏ lẻ). | `"0"`, `"1"` |
| `aor_id` | Chuỗi ID số | Mã quận / khu vực thương mại đặt quán (từ 0 đến 10). | `"0"`, `"6"` |
| `poi_score` | Số thực (float) | Điểm đánh giá tổng thể của quán (thang điểm 5.0). | `4.66`, `4.79` |
| `delivery_comment_avg_score` | Số thực (float) | Điểm đánh giá chất lượng dịch vụ giao hàng. | `4.87`, `4.97` |
| `food_comment_avg_score` | Số thực (float) | Điểm đánh giá chất lượng đồ ăn của quán. | `4.61`, `4.82` |

---

### 1.3. Bảng thông tin món ăn (`spus.txt`)
* **Tổng số dòng:** 179.778 dòng (179.759 món ăn SPU duy nhất).

| Tên trường (Cột) | Kiểu dữ liệu | Ý nghĩa tiếng Việt | Ví dụ thực tế |
| :--- | :--- | :--- | :--- |
| `wm_food_spu_id` | Chuỗi ID số | Mã định danh duy nhất của món ăn. | `"142475"`, `"168385"` |
| `wm_food_spu_name` | Chuỗi ký tự | Tên món ăn (đã được mã hóa). | `"a22084-de2424-..."` |
| `price` | Số thực (float) | Giá tiền món ăn (đơn vị: Nhân dân tệ - CNY). | `32.0`, `8.0`, `15.5` |
| `category` | Chuỗi mảng | Nhóm phân loại món ăn. | `"[0]"`, `"[1]"` |
| `ingredients` | Chuỗi | Danh sách nguyên liệu chế biến món ăn. | `"NULL"` hoặc mã hóa |
| `taste` | Chuỗi | Khẩu vị / hương vị món ăn (cay, chua, ngọt...). | `"NULL"` hoặc mã hóa |
| `stand_food_id` | Chuỗi ID số | Mã định danh món ăn theo danh mục chuẩn hóa. | `"0"`, `"NULL"` |
| `stand_food_name` | Chuỗi | Tên món ăn chuẩn hóa. | `"NULL"` hoặc mã hóa |

---

### 1.4. Bảng đơn hàng 21 ngày đầu (`orders_train.txt`)
* **Tổng số dòng:** 1.068.495 đơn hàng trong giai đoạn Huấn luyện (từ `20210301` đến `20210321`).

| Tên trường (Cột) | Kiểu dữ liệu | Ý nghĩa tiếng Việt | Ví dụ thực tế |
| :--- | :--- | :--- | :--- |
| `user_id` | Chuỗi ID số | Mã người dùng thực hiện đơn hàng. | `"140760"` |
| `wm_order_id` | Chuỗi ID số | Mã định danh duy nhất của đơn hàng. | `"21559"`, `"1068495"` |
| `wm_poi_id` | Chuỗi ID số | Mã quán ăn được khách chốt đơn đặt món. | `"1782"`, `"2198"` |
| `aor_id` | Chuỗi ID số | Mã quận / khu thương mại nơi diễn ra đơn hàng (0 đến 10). | `"0"`, `"6"` |
| `order_price_interval` | Chuỗi phân khoảng | Khoảng tổng giá trị của đơn hàng. | `"<29"`, `"[36,49)"`, `">=65"` |
| `order_timestamp` | Số nguyên (int) | Mốc thời gian đặt đơn dạng Unix Epoch (tính bằng giây). | `1614524412` |
| `ord_period_name` | Chuỗi số (0 - 4)| Khung giờ đặt đơn trong ngày:<br>• `0`: Sáng sớm / Bữa sáng<br>• `1`: Bữa trưa (10h - 14h)<br>• `2`: Trà chiều (14h - 17h)<br>• `3`: Bữa tối (17h - 21h)<br>• `4`: Bữa đêm (21h - sáng hôm sau) | `"1"` (Bữa trưa), `"3"` (Bữa tối) |
| `order_scene_name` | Chuỗi ID số | Mã định danh bối cảnh đặt hàng (ở nhà, công ty, sự kiện...). | `"0"`, `"1"` |
| `aoi_id` | Chuỗi ID số | Mã cụm địa chỉ giao hàng cụ thể của khách (3.551 địa chỉ). | `"169"`, `"123"` |
| `takedlvr_aoi_type_name` | Chuỗi ID số | Loại hình khu vực nhận hàng (tòa văn phòng, chung cư, trường học...). | `"1"`, `"2"` |
| `dt` | Chuỗi (`YYYYMMDD`)| Ngày đặt đơn hàng. | `"20210301"` đến `"20210321"` |

---

### 1.5. Bảng món ăn trong đơn hàng (`orders_spu_train.txt`)
* **Tổng số dòng:** 3.445.180 dòng (trung bình 1 đơn hàng chứa ~3.2 món ăn).

| Tên trường (Cột) | Kiểu dữ liệu | Ý nghĩa tiếng Việt | Ví dụ thực tế |
| :--- | :--- | :--- | :--- |
| `wm_order_id` | Chuỗi ID số | Mã đơn hàng (khóa ngoại liên kết sang `orders_train.txt`). | `"43925"` |
| `wm_food_spu_id` | Chuỗi ID số | Mã món ăn cụ thể được khách chọn mua trong đơn hàng này. | `"142475"` |
| `dt` | Chuỗi (`YYYYMMDD`)| Ngày đặt đơn. | `"20210301"` |

---

### 1.6. Bảng chuỗi click xem quán trước khi đặt (`orders_poi_session.txt`)
* **Tổng số dòng:** 1.299.045 dòng phiên duyệt quán.

| Tên trường (Cột) | Kiểu dữ liệu | Ý nghĩa tiếng Việt | Ví dụ thực tế |
| :--- | :--- | :--- | :--- |
| `wm_order_id` | Chuỗi ID số | Mã đơn hàng phát sinh ngay sau phiên duyệt này. | `"21559"` |
| `clicks` | Chuỗi ký tự | Chuỗi các quán ăn khách đã bấm vào xem liên tục ngay trước khi đặt món, các quán phân tách nhau bởi dấu `#`. Nếu khách vào đặt ngay không lướt thì có giá trị `NULL`. | `"1782#535#1002#26705"` hoặc `"NULL"` |
| `dt` | Chuỗi (`YYYYMMDD`)| Ngày diễn ra phiên tương tác. | `"20210301"` |

---

### 1.7. Dữ liệu tuần kiểm thử thứ 4 (Test: `20210322` – `20210328`)
Tuần 4 được chia thành 2 nhóm file riêng biệt để phục vụ đánh giá mô hình:
1. **Thông tin đơn hàng tuần 4:**
   * `orders_test_poi.txt` (230.550 dòng): Chứa bối cảnh đơn hàng (`user_id`, `wm_order_id`, `aor_id`, `order_timestamp`, `ord_period_name`, `aoi_id`, `takedlvr_aoi_type_name`, `dt`), **không có cột `wm_poi_id`**.
   * `orders_poi_test_label.txt` (230.550 dòng): Chứa nhãn quán thực tế khách đã đặt (`user_id`, `wm_order_id`, `wm_poi_id`, `dt`).
2. **Thông tin món ăn tuần 4:**
   * `orders_test_spu.txt` (752.447 dòng): Bối cảnh gọi món, không có mã món.
   * `orders_spu_test_label.txt` (752.447 dòng): Nhãn món ăn thực tế khách đã chọn (`user_id`, `wm_order_id`, `wm_food_spu_id`, `dt`).

---

## 2. PRETRAINING: ĐẦU VÀO + ĐẦU RA (USER FOUNDATION MODEL)

* **Mục tiêu:** Huấn luyện mô hình Transformer hiểu thói quen, mức chi tiêu và sở thích dài hạn/ngắn hạn của người dùng từ toàn bộ chuỗi hành vi lịch sử trong 21 ngày đầu (`20210301` đến `20210321`).
* **Thời điểm cắt dữ liệu (`cutoff_time`):** `2021-03-21 23:59:59`.

### 2.1. Đầu vào (Input)
Mỗi mẫu dữ liệu huấn luyện đại diện cho một người dùng, gồm 2 khối dữ liệu:

1. **Khối đặc trưng bảng tĩnh (`tabular`):**
   * `home_district`: Mã quận quen thuộc nhất của user (quận `aor_id` mà user đặt nhiều đơn nhất trong 21 ngày).
   * `order_count_7d`: Tổng số đơn hàng user đã đặt trong 7 ngày gần nhất (từ ngày `20210315` đến `20210321`).
   * `avg_pay_amt`: Mức chi tiêu trung bình lịch sử lấy từ `users.txt` (mã hóa categorical 5 mức: 0: `<29`, 1: `[29,36)`, 2: `[36,49)`, 3: `[49,65)`, 4: `>=65`).
   * `preferred_category`: Ngành hàng quán ăn ưa thích nhất (danh mục `primary_first_tag_name` có tần suất tương tác cao nhất).
2. **Khối chuỗi hành vi lịch sử (`history`):**
   * Chuỗi từ 20 đến 50 hành vi gần nhất tính đến hết ngày 21 được sắp xếp theo thời gian tăng dần (`order_timestamp`).
   * Mỗi bước hành vi $t$ bao gồm:
     * `action`: Loại hành động (`view_merchant` nếu lấy từ chuỗi `clicks` trong `orders_poi_session.txt`, hoặc `order_merchant` nếu là hành vi chốt đơn quán trong `orders_train.txt`).
     * `wm_poi_id`: Mã định danh quán ăn tương tác.
     * `ord_period_name`: Khung giờ diễn ra hành vi (0 đến 4).
     * `timestamp`: Mốc thời gian Unix Epoch của hành vi.

### 2.2. Đầu ra (Output)
1. **Trong quá trình huấn luyện (Training Loss):**
   * Hàm mất mát Cross-Entropy cho hai tác vụ tự giám sát:
     * *Masked Item Reconstruction:* Che ngẫu nhiên 15% mã quán trong chuỗi và bắt mô hình khôi phục đúng quán bị che.
     * *Next-item Prediction:* Dựa vào chuỗi hành vi từ bước $1 \dots t$, dự đoán mã quán ở bước $t+1$.
2. **Khi suy luận thực tế (Inference Representation):**
   * Trích xuất ra **2 vector biểu diễn người dùng**:
     * `long_embedding`: Vector số thực đại diện cho sở thích cố định, tính cách và năng lực chi tiêu dài hạn của người dùng (sinh từ User ID và khối Tabular).
     * `short_embedding`: Vector số thực đại diện cho ý định ẩm thực và nhu cầu tức thời gần đây (tổng hợp qua lớp Attention/Pooling từ trạng thái ẩn của chuỗi `history`).

---

## 3. DOWNSTREAM 1: ĐẦU VÀO + ĐẦU RA (GỢI Ý MÓN ĂN / QUÁN ĂN)

* **Mục tiêu:** Dự đoán xác suất người dùng bấm vào xem (`clicked`) và đặt món (`ordered`) khi một quán ăn/món ăn xuất hiện trên màn hình ứng dụng.
* **Quy ước dữ liệu:** Sử dụng dữ liệu tuần thứ 4 (`orders_test_poi.txt` ghép với `orders_poi_test_label.txt`).
  * Tập Train: Ngày `20210322` đến `20210325`.
  * Tập Test: Ngày `20210326` đến `20210328`.

### 3.1. Đầu vào (Input)
Ghép nối các vector đặc trưng của người dùng, quán ăn và bối cảnh tại thời điểm phát sinh yêu cầu:
1. **Đặc trưng người dùng:**
   * Cặp vector biểu diễn `[long_embedding, short_embedding]` được trích xuất từ User Foundation Model ở Phần 2.
2. **Đặc trưng quán ăn / món ăn ứng viên (`item`):**
   * `wm_poi_id`: Mã định danh quán ăn ứng viên.
   * `primary_first_tag_name`: Nhóm ngành hàng của quán.
   * `poi_score`, `food_comment_avg_score`: Điểm đánh giá sao của quán và điểm món ăn.
   * `price`: Giá món ăn (từ `spus.txt`) hoặc khoảng giá đơn hàng (`order_price_interval`).
   * `delivery_fee`: Phí giao hàng quy ước (15.000 VND nếu cùng quận, 25.000 VND nếu khác quận).
   * `estimated_delivery_minutes`: Thời gian giao hàng dự kiến, quy ước cố định theo quán: $20 + (\text{wm\_poi\_id} \pmod{15})$ phút.
3. **Đặc trưng bối cảnh (`context`):**
   * `aor_id`: Khu vực giao hàng (mã quận).
   * `aoi_id`: Cụm địa chỉ giao hàng cụ thể.
   * `ord_period_name`: Khung giờ đặt đơn (0 đến 4).
   * `local_hour`: Giờ trong ngày (0 đến 23) trích xuất từ `order_timestamp`.

### 3.2. Đầu ra (Output)
1. **Giá trị dự đoán:** Hai xác suất nằm trong đoạn $[0.0, 1.0]$:
   * $P(\text{click})$: Xác suất người dùng bấm vào xem quán/món ăn.
   * $P(\text{order})$: Xác suất người dùng thực sự chốt đơn quán/món ăn đó.
2. **Cơ chế xếp hạng (Ranking):**
   * Điểm xếp hạng tổng hợp = $P(\text{click}) \times P(\text{order})$.
   * Hệ thống sắp xếp các quán/món ăn ứng viên theo điểm từ cao xuống thấp và đề xuất Top-5 quán tốt nhất.
3. **Nhãn huấn luyện (Ground-truth Labels):**
   * Mẫu dương (Positive): Quán nằm trong chuỗi `clicks` $\rightarrow$ `clicked = 1`; quán được chốt đơn thanh toán $\rightarrow$ `ordered = 1`.
   * Mẫu âm (Negative): Chọn ngẫu nhiên 3–5 quán khác cùng quận mà người dùng không tương tác trong phiên đó $\rightarrow$ `clicked = 0, ordered = 0`.
4. **Chỉ số đánh giá (Evaluation Metric):**
   * Đo đạc bằng **NDCG@5** và **Recall@5** trên tập kiểm thử (ngày 26 đến 28).
   