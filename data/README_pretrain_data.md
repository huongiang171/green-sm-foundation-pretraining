# BỘ DỮ LIỆU TIỀN HUẤN LUYỆN NGƯỜI DÙNG (OMNIBEHAVIOR PRETRAIN V2) & THIẾT KẾ CẤU TRÚC DỮ LIỆU

---

## 1. THÔNG TIN BỘ DỮ LIỆU & GIẢI NGHĨA CÁC TRƯỜNG THỰC TẾ

Bộ dữ liệu **OmniBehavior User Pretraining Observations (v2)** (`hongduc05/pretrain_gsm_usr`) là tập dữ liệu quan sát hành vi người dùng đa kịch bản (Multi-scenario Human Behavior) được thiết kế chuyên biệt cho giai đoạn **tiền huấn luyện mô hình nền tảng người dùng (User Foundation Model)**.
Bộ dữ liệu được kế thừa và chuẩn hóa từ nghiên cứu *OmniBehavior: Towards Real-world Human Behavior Simulation* (Chen et al., 2026), kết hợp cùng kiến trúc biểu diễn hành vi người dùng lấy cảm hứng từ *User Foundation Models for Grab*.

* **Định dạng lưu trữ:** Apache Parquet (nén bằng thuật toán Zstandard - `zstd` level 3), schema chuẩn hóa Apache Arrow 24.0.0.
* **Thời gian thu thập & Phân chia:** Dữ liệu kéo dài 3 tháng (từ `01/09/2025` đến `30/11/2025`).
* **Quy mô tổng thể:**
  * **186 người dùng** độc lập (Users).
  * **804.435** bản ghi nguồn (raw interaction records).
  * **891.565** quan sát hành động đơn lẻ (action observations / events).
  * **792.481** điểm ảnh chụp trạng thái (snapshots) phục vụ huấn luyện mô hình.
  * Tổng dung lượng lưu trữ Parquet: **~550 MB** (Events: 514,36 MB, Snapshots: 35,89 MB).

### Phân bố Logical Split theo thời gian:
Bộ dữ liệu được chia theo trục thời gian thực (Time-based Splitting) để ngăn chặn tuyệt đối hiện tượng rò rỉ dữ liệu tương lai (Data Leakage):

| Logical Split | Khoảng thời gian hiệu lực | Số lượng Snapshots | Tỷ lệ | Mục đích |
| :--- | :--- | :---: | :---: | :--- |
| **Train** | 01/09/2025 – 31/10/2025 (2 tháng đầu) | **531.852** | 67,1% | Huấn luyện tiền nhận thức (Pretraining Representation) |
| **Validation** | 01/11/2025 – 15/11/2025 (nửa đầu tháng 11) | **131.582** | 16,6% | Tinh chỉnh siêu tham số và checkpoint dừng sớm |
| **Test** | 16/11/2025 – 30/11/2025 (nửa cuối tháng 11) | **129.047** | 16,3% | Đánh giá năng lực mô hình trên dữ liệu tương lai |

---

### 1.1. Bảng sự kiện chi tiết (`events/{user_id}.parquet`)
* **Tổng số dòng:** 891.565 dòng trên 186 files tương ứng 186 user.
* **Đơn vị bản ghi:** Mỗi dòng là **một hành động quan sát đơn lẻ (Action Observation)**, bao gồm cả trạng thái hành động thành công và trạng thái âm (chưa hoàn tất/hủy).

| Tên trường (Cột) | Kiểu dữ liệu | Ý nghĩa tiếng Việt | Ví dụ thực tế |
| :--- | :--- | :--- | :--- |
| `row_index` | `int32` | Thứ tự chỉ mục dòng của sự kiện trong file parquet của user đó. | `0`, `1`, `136` |
| `event_id` | `string` | Mã định danh duy nhất của sự kiện: `{user_id}:{source_record_index}:{action_index}`. | `"user_001:0:0"`, `"user_014:12:1"` |
| `user_id` | `string` | Mã định danh người dùng thực hiện hành vi. | `"user_001"`, `"user_089"` |
| `source_record_id` | `string` | Mã định danh bản ghi nguồn `{user_id}:{source_record_index}` liên kết các hành động cùng ngữ cảnh. | `"user_001:0"` |
| `source_record_index` | `int32` | Chỉ mục của bản ghi tương tác gốc trong log của người dùng. | `0`, `5`, `120` |
| `action_index` | `int32` | Thứ tự của hành động cụ thể bên trong cùng một bản ghi nguồn. | `0`, `1` |
| `source_action_count` | `int32` | Tổng số lượng hành động phát sinh trong cùng một bản ghi nguồn đó. | `1`, `2`, `4` |
| `source_timestamp` | `string` | Mốc thời gian gốc được ghi nhận từ hệ thống (có thể chỉ có ngày hoặc có giờ phút giây). | `"2025-09-02"`, `"2025-09-19T02:37:25"` |
| `event_time` | `string` | Thời gian xảy ra sự kiện định dạng chuẩn hóa ISO. | `"2025-09-02"`, `"2025-09-19T02:37:25"` |
| `time_precision` | `string` | Độ chính xác thời gian: `"second"` (chính xác đến giây) hoặc `"day"` (chỉ có ngày). | `"day"`, `"second"` |
| `time_lower_bound` | `timestamp[us]` | Mốc cận dưới thời gian của sự kiện (Microsecond timestamp). | `2025-09-02 00:00:00.000000` |
| `time_upper_bound` | `timestamp[us]` | Mốc cận trên thời gian của sự kiện. Với sự kiện chỉ có ngày D, cận trên là đầu ngày D+1. | `2025-09-03 00:00:00.000000` |
| `upper_bound_exclusive`| `bool` | Cờ xác định cận trên là khoảng mở (`true` nếu là ngày `[D, D+1)`) hay khoảng đóng (`false`). | `true`, `false` |
| `eligibility_time` | `timestamp[us]` | Thời điểm sự kiện đủ điều kiện (eligible) xuất hiện trong snapshot mà không bị rò rỉ dữ liệu. | `2025-09-03 00:00:00.000000` |
| `eligibility_inclusive`| `bool` | Cờ xác định tính bao gồm tại mốc thời gian đủ điều kiện. | `false`, `true` |
| `scenario` | `string` | Kịch bản tương tác ứng dụng (1 trong 6 kịch bản lớn): `Advertisement`, `Live Streaming`, `Video Browsing`, `E-commerce`, `Customer Service`, `Search Behavior`. | `"Video Browsing"`, `"E-commerce"` |
| `event_type` | `string` | Loại hành động cụ thể: `watch`, `like`, `comment`, `purchase`, `cart`, `conversion`, `dialogue`, `search`, `follow`,... | `"watch"`, `"purchase"`, `"conversion"` |
| `context_json` | `large_string` | Chuỗi JSON chứa toàn bộ ngữ cảnh gốc của môi trường tương tác (sản phẩm, tiêu đề, thời lượng video,...). | `{"product": "Peach", "price": 29.5}` |
| `action_json` | `large_string` | Chuỗi JSON chứa thông số hành động cụ thể và các chỉ số đo lường (kèm theo cờ phủ định nếu hành vi thất bại). | `{"type": "conversion", "conversion_cnt": 2}` |

---

### 1.2. Bảng ảnh chụp lịch sử & Profile (`snapshots/{user_id}.parquet`)
* **Tổng số dòng:** 792.481 dòng trên 186 files.
* **Vai trò:** Mỗi dòng đại diện cho một mẫu dữ liệu huấn luyện độc lập (Training Sample / Snapshot) của người dùng tại thời điểm cắt `cutoff_time`.

| Tên trường (Cột) | Kiểu dữ liệu | Ý nghĩa tiếng Việt | Ví dụ thực tế |
| :--- | :--- | :--- | :--- |
| `sample_id` | `string` | Mã định danh duy nhất của snapshot: `{user_id}:{cutoff_time}:{open/closed}`. | `"user_001:2025-09-03T00:00:00:open"` |
| `user_id` | `string` | Mã người dùng sở hữu snapshot này. | `"user_001"` |
| `cutoff_time` | `timestamp[us]` | Mốc thời gian cắt dữ liệu của snapshot. Tuyệt đối không chứa thông tin sau mốc này. | `2025-09-03 00:00:00.000000` |
| `cutoff_inclusive` | `bool` | Cờ quy định snapshot cắt đóng (`true`) hay cắt mở (`false`) tại điểm thời gian. | `false`, `true` |
| `split` | `string` | Logical Split mà snapshot thuộc về: `"train"`, `"validation"`, hoặc `"test"`. | `"train"`, `"validation"`, `"test"` |
| `history_start` | `int32` | Vị trí bắt đầu của chuỗi sự kiện lịch sử trong file `events/{user_id}.parquet`. | `0`, `25`, `107` |
| `history_stop` | `int32` | Vị trí kết thúc của chuỗi sự kiện lịch sử (chỉ số nửa mở, tối đa lấy 30 sự kiện gần nhất). | `1`, `30`, `137` |
| `history_event_ids` | `list<string>` | Danh sách các `event_id` nằm trong chuỗi lịch sử của snapshot này. | `["user_001:0:0"]` |
| `history_order_known` | `bool` | Cờ chỉ định thứ tự thời gian giữa các event trong chuỗi có được xác định chắc chắn hay không. | `true`, `false` |
| `history_truncated` | `bool` | Cờ báo hiệu lịch sử thực tế của user vượt quá giới hạn 30 events và đã bị cắt ngắn về 30. | `false`, `true` |
| `history_group_truncated` | `bool` | Cờ báo hiệu việc cắt ngắn 30 events đã cắt đôi một nhóm hành động có cùng timestamp. | `false`, `true` |
| `profile_json` | `string` | Chuỗi JSON chứa 9 chỉ số hồ sơ thống kê tĩnh của user trong cửa sổ 30 ngày trước cutoff. | `{"active_days_30d": 1, ...}` |

---

### 1.3. Khối hồ sơ người dùng 30 ngày (`input.profile`)
Khối hồ sơ được tính toán bằng cách tổng hợp toàn bộ các quan sát hợp lệ trong khoảng `(cutoff_time - 30 ngày, cutoff_time]`. Khối này gồm **9 đặc trưng số học (numerical)** phản ánh mức độ hoạt động và phân bổ mối quan tâm của người dùng:

| Tên trường | Kiểu dữ liệu | Ý nghĩa tiếng Việt | Khoảng giá trị & Ví dụ |
| :--- | :--- | :--- | :--- |
| `as_of_time` | `string ISO` | Mốc thời gian hiệu lực của bản profile (trùng với `cutoff_time`). | `"2025-09-03T00:00:00"` |
| `active_days_30d` | `int` | Số ngày lịch người dùng có phát sinh ít nhất một tương tác trong 30 ngày qua. | `1` đến `31` ngày |
| `event_count_30d` | `int` | Tổng số lượng hành động quan sát được của người dùng trong 30 ngày qua. | `1` đến hàng ngàn events |
| `video_browsing_event_count_30d` | `int` | Số lượng tương tác trong kịch bản **Lướt Video ngắn** (chiếm ~77.5% toàn dataset). | `0`, `54`, `320` |
| `live_streaming_event_count_30d` | `int` | Số lượng tương tác trong kịch bản **Xem Livestream** (chiếm ~17.0% toàn dataset). | `0`, `12`, `85` |
| `e_commerce_event_count_30d` | `int` | Số lượng tương tác trong kịch bản **Thương mại điện tử / Mua sắm** (~2.9%). | `0`, `3`, `15` |
| `advertisement_event_count_30d` | `int` | Số lượng tương tác với **Quảng cáo hiển thị** (~0.7%). | `0`, `1`, `5` |
| `search_behavior_event_count_30d` | `int` | Số lượng hành vi **Tìm kiếm từ khóa** (~1.6%). | `0`, `2`, `8` |
| `customer_service_event_count_30d` | `int` | Số lượng phiên tương tác với **Chăm sóc khách hàng** (~0.2%). | `0`, `1` |

---

### 1.4. Khối ngữ cảnh hành vi (`context`) theo 6 Scenario (Allowlist)
Để loại trừ các trường có nguy cơ gây rò rỉ dữ liệu (như doanh số 180 ngày, kết quả sau khi mua), bộ dữ liệu áp dụng danh sách trường cho phép (**Allowlist**) nghiêm ngặt cho từng kịch bản:

| Kịch bản (`scenario`) | Các trường ngữ cảnh được phép sử dụng trong `input.context` | Ý nghĩa nghiệp vụ |
| :--- | :--- | :--- |
| **Advertisement** | `product`, `industry_main`, `industry_sub`, `caption`, `ocr_text`, `asr_text` | Tên sản phẩm quảng cáo, ngành hàng chính/phụ, văn bản mô tả, chữ nhận diện từ ảnh/video (OCR), âm thanh nhận diện (ASR). |
| **Live Streaming** | `live_category`, `live_title`, `is_shop_live`, `live_cover_content` | Thể loại phòng live, tiêu đề live, cờ có bán hàng hay không (`true`/`false`), nội dung ảnh bìa. |
| **Video Browsing** | `caption`, `photo_type`, `duration`, `ocr`, `asr` | Chú thích video, thể loại ảnh/video, thời lượng phát video (giây), văn bản nhận dạng OCR và ASR. |
| **E-commerce** | `product_source`, `product_id`, `item_title`, `item_desc`, `item_price`, `category_level1_name`, `category_level2_name`, `category_level3_name`, `industry_level1_name`, `industry_level2_name`, `industry_level3_name` | Mã nguồn sản phẩm, mã hàng hóa, tiêu đề, mô tả sản phẩm, giá bán, cây phân cấp ngành hàng 3 cấp (`level1` đến `level3`). |
| **Customer Service**| `ticket_category`, `product_name`, `item_price`, `product_category_info` | Thể loại khiếu nại/hỗ trợ, tên sản phẩm liên quan, giá tiền và thông tin nhóm hàng hóa. |
| **Search Behavior** | *(Không sử dụng context)* | Kịch bản tìm kiếm chỉ ghi nhận từ khóa và hành động click/show. |

---

### 1.5. Khối hành động thực tế (`action`) & Thống kê Action Types
Mỗi sự kiện mang một đối tượng `action` phản ánh chính xác hành vi của người dùng và các thuộc tính đo lường:

| Nhóm hành động | Tên `event_type` | Tần suất xuất hiện | Các trường dữ liệu chính trong `action` |
| :--- | :--- | :---: | :--- |
| **Theo dõi nội dung** | `watch` | **781.921** (87,7%) | `play_duration` (thời gian xem thực tế), `progress` (tiến độ xem), `complete_play` (xem hết video hay không). |
| **Tương tác xã hội** | `like`, `comment`, `share`, `collect`, `dislike` | **54.796** (6,1%) | `comment_text` (nội dung bình luận), `is_like`, `share_type`. |
| **Giao dịch TMĐT** | `purchase`, `cart`, `click_cart` | **29.240** (3,3%) | `item_cnt` (số lượng mua), `order_amt` (tổng tiền), **`is_pay`** (cờ đã thanh toán hay hủy), `sku_id`. |
| **Tìm kiếm & Quảng cáo** | `search`, `show`, `click`, `conversion` | **18.094** (2,0%) | `query` (từ khóa tìm kiếm), `conversion_cnt` (số lần chuyển đổi), `click_cnt`. |
| **Quan hệ & Khác** | `follow`, `unfollow`, `send_gift`, `activation`, `dialogue` | **7.514** (0,9%) | `gift_value` (giá trị quà), `dialogue_turns` (số lượt chat), `status`. |

> [!IMPORTANT]
> **Bảo toàn trạng thái âm (Negative States):** Các hành động như `purchase` với cờ `is_pay=false` hoặc `conversion` với `conversion_cnt=0` không bị lọc bỏ mà được giữ nguyên vẹn để mô hình học được cả tín hiệu tiêu cực (người dùng từ bỏ giỏ hàng, hủy đơn).

---

### 1.6. Cấu trúc JSON thực tế của dữ liệu (Data JSON Schema & Samples)

Dưới đây là cấu trúc đối tượng JSON hoàn chỉnh của một mẫu snapshot tái cấu trúc (reconstructed snapshot record) được trả về bởi hàm `iter_pretraining_samples()` hoặc lưu trong `preview.json`. 

Mỗi bản ghi đại diện cho một mẫu huấn luyện của một người dùng tại một thời điểm cắt cụ thể (`cutoff_time`), gồm 2 phần chính: `sample_id` và khối `input`.

#### 1.6.1. Mẫu JSON đầy đủ của một Snapshot Record

```json
{
  "sample_id": "user_001:2025-09-19T02:37:25:closed",
  "input": {
    "user_id": "user_001",
    "cutoff_time": "2025-09-19T02:37:25",
    "cutoff_inclusive": true,
    "timezone": null,
    "history_order_known": false,
    "history_truncated": false,
    "history_group_truncated": false,
    "profile": {
      "as_of_time": "2025-09-19T02:37:25",
      "active_days_30d": 5,
      "event_count_30d": 13,
      "video_browsing_event_count_30d": 0,
      "live_streaming_event_count_30d": 1,
      "e_commerce_event_count_30d": 0,
      "advertisement_event_count_30d": 12,
      "search_behavior_event_count_30d": 0,
      "customer_service_event_count_30d": 0
    },
    "history": [
      {
        "event_id": "user_001:0:0",
        "source_record_id": "user_001:0",
        "source_record_index": 0,
        "action_index": 0,
        "source_action_count": 1,
        "event_time": "2025-09-02",
        "event_type": "conversion",
        "scenario": "Advertisement",
        "time_precision": "day",
        "time_lower_bound": "2025-09-02T00:00:00",
        "time_upper_bound": "2025-09-03T00:00:00",
        "upper_bound_exclusive": true,
        "eligibility_time": "2025-09-03T00:00:00",
        "eligibility_inclusive": false,
        "context": {
          "product": "The peaches have arrived.",
          "industry_main": "Food and Beverage",
          "industry_sub": "Condiments",
          "caption": null,
          "ocr_text": null,
          "asr_text": null
        },
        "action": {
          "type": "conversion",
          "conversion_cnt": 2
        }
      },
      {
        "event_id": "user_001:10:0",
        "source_record_id": "user_001:10",
        "source_record_index": 10,
        "action_index": 0,
        "source_action_count": 1,
        "event_time": "2025-09-19T02:37:25",
        "event_type": "watch",
        "scenario": "Live Streaming",
        "time_precision": "second",
        "time_lower_bound": "2025-09-19T02:37:25",
        "time_upper_bound": "2025-09-19T02:37:25",
        "upper_bound_exclusive": false,
        "eligibility_time": "2025-09-19T02:37:25",
        "eligibility_inclusive": true,
        "context": {
          "live_category": "Other",
          "live_title": "Mr. Xin is leaving",
          "is_shop_live": true,
          "live_cover_content": "Live stream title and banner"
        },
        "action": {
          "type": "watch",
          "watch_time": 45
        }
      }
    ]
  }
}
```

#### 1.6.2. Cấu trúc JSON chi tiết của Event trong từng Kịch bản (Scenario)

Tùy theo kịch bản tương tác (`scenario`), đối tượng `context` và `action` bên trong mỗi event sẽ có các trường dữ liệu cụ thể:

##### 1. Kịch bản Lướt Video ngắn (`Video Browsing`) - Chiếm 77.5%
```json
{
  "event_id": "user_001:11:0",
  "event_time": "2025-09-19T10:36:38",
  "scenario": "Video Browsing",
  "event_type": "watch",
  "time_precision": "second",
  "context": {
    "caption": "#PureHandmadeBeanThreadNoodles #TrendingNow",
    "photo_type": "VIDEO",
    "duration": 563,
    "ocr": "Selling fake vermicelli will leave you without descendants...",
    "asr": "Văn bản nhận diện giọng nói trong video..."
  },
  "action": {
    "type": "watch",
    "play_duration": 42000,
    "complete_play": true
  }
}
```

##### 2. Kịch bản Xem Livestream (`Live Streaming`) - Chiếm 17.0%
```json
{
  "event_id": "user_001:10:0",
  "event_time": "2025-09-19T02:37:25",
  "scenario": "Live Streaming",
  "event_type": "watch",
  "time_precision": "second",
  "context": {
    "live_category": "Culinary & Cooking",
    "live_title": "Nấu ăn gia đình cùng đầu bếp",
    "is_shop_live": true,
    "live_cover_content": "Banner khuyến mãi phòng live"
  },
  "action": {
    "type": "watch",
    "watch_time": 185
  }
}
```

##### 3. Kịch bản Mua sắm & Thương mại điện tử (`E-commerce`) - Chiếm 2.9%
```json
{
  "event_id": "user_001:13:0",
  "event_time": "2025-09-20T08:37:24",
  "scenario": "E-commerce",
  "event_type": "cart",
  "time_precision": "second",
  "context": {
    "product_source": "Order List Page",
    "item_title": "Bình giữ nhiệt cầm tay 1000ml inox 304",
    "item_desc": null,
    "item_price": 45.0,
    "category_level1_name": "Đồ gia dụng",
    "category_level2_name": "Dụng cụ nhà bếp",
    "category_level3_name": "Bình nước giữ nhiệt",
    "industry_level1_name": "Đời sống hàng ngày",
    "industry_level2_name": "Đồ dùng sinh hoạt",
    "industry_level3_name": "Thiết bị ăn uống"
  },
  "action": {
    "type": "cart",
    "is_add_to_cart": true
  }
}
```

##### 4. Kịch bản Tìm kiếm (`Search Behavior`) - Chiếm 1.6%
```json
{
  "event_id": "user_002:24:0",
  "event_time": "2025-09-01T00:00:57",
  "scenario": "Search Behavior",
  "event_type": "search",
  "time_precision": "second",
  "context": null,
  "action": {
    "type": "search",
    "keyword": "Wild Leopard Cue Authenticity",
    "query_category": "Browsing Type"
  }
}
```

##### 5. Kịch bản Quảng cáo (`Advertisement`) - Chiếm 0.7%
```json
{
  "event_id": "user_001:0:0",
  "event_time": "2025-09-02",
  "scenario": "Advertisement",
  "event_type": "conversion",
  "time_precision": "day",
  "context": {
    "product": "The peaches have arrived.",
    "industry_main": "Food and Beverage",
    "industry_sub": "Condiments",
    "caption": "Quảng cáo đào tươi và gia vị",
    "ocr_text": null,
    "asr_text": null
  },
  "action": {
    "type": "conversion",
    "conversion_cnt": 2
  }
}
```

##### 6. Kịch bản Chăm sóc khách hàng (`Customer Service`) - Chiếm 0.2%
```json
{
  "event_id": "user_001:57:0",
  "event_time": "2025-11-02T12:37:19",
  "scenario": "Customer Service",
  "event_type": "dialogue",
  "time_precision": "second",
  "context": {
    "ticket_category": "Product Inquiry",
    "product_name": "Tuýp kem dưỡng da phục hồi kháng khuẩn",
    "item_price": 99.9,
    "product_category_info": "Chăm sóc da & Y tế"
  },
  "action": {
    "type": "dialogue",
    "turns": 3
  }
}
```

---

## 2. PRETRAINING: ĐẦU VÀO + ĐẦU RA (USER FOUNDATION MODEL)

* **Mục tiêu:** Huấn luyện mô hình Transformer hiểu sâu sắc chân dung dài hạn (sở thích tĩnh, thói quen sinh hoạt) và ý định ngắn hạn (mối quan tâm tức thời) của người dùng từ các luồng hành vi đa kịch bản.
* **Thời điểm cắt dữ liệu (`cutoff_time`):** Tạo mẫu động sau mỗi mốc eligibility coordinate, đảm bảo không có rò rỉ dữ liệu tương lai.

```
                           ┌────────────────────────────┐
                           │   Raw Parquet Data Store   │
                           │ (events/ & snapshots/)     │
                           └─────────────┬──────────────┘
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
         [Khối Tabular Profile]                     [Khối History Sequence]
         - active_days_30d                          - Tối đa 30 events gần nhất
         - event_count_30d                          - scenario_ids & event_type_ids
         - 6 scenario counts                        - Temporal Interval Encoding
                   │                                           │
                   ▼                                           ▼
         ┌───────────────────┐                      ┌────────────────────┐
         │  Tabular Projector│                      │ Sequence Embedding │
         └─────────┬─────────┘                      └──────────┬─────────┘
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                            ┌────────────────────────┐
                            │   Shared Transformer   │
                            │   Encoder (d=128)      │
                            └────────────┬───────────┘
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
         [Self-Supervised Losses]                    [User Representations]
         - Masked Reconstruction Loss                - z_long  (Long-term Embedding)
         - Next-Action Prediction Loss               - z_short (Short-term Embedding)
                                                     - z_user  = Fusion(z_long, z_short)
```

### 2.1. Đầu vào (Input)
Mỗi mẫu dữ liệu huấn luyện gồm 3 phần:
1. **User ID:** Mã định danh người dùng đưa qua **User ID Adapter** (lookup embedding layer) để học các đặc trưng cá nhân hóa dài hạn.
2. **Khối Profile người dùng (`profile`):** Vector đặc trưng tĩnh 8 chiều chuẩn hóa qua hàm $\log(1 + x)$ hoặc token hóa dạng cặp `key:value` (`active_days:15`, `video_count:120`,...).
3. **Khối chuỗi sự kiện lịch sử (`history`):** Chuỗi tối đa **30 sự kiện gần nhất** tính đến `cutoff_time`. Mỗi bước thời gian $t$ bao gồm:
   - `scenario_id`: Mã hóa nhúng của 1 trong 6 kịch bản.
   - `event_type_id`: Mã hóa nhúng của loại hành vi (`watch`, `like`, `cart`, `purchase`,...).
   - `temporal_features`: Khoảng thời gian từ lúc phát sinh đến cutoff, độ chính xác thời gian (`second` / `day`).
   - `context_tokens`: Các token ngữ cảnh (tên ngành hàng, tên sản phẩm).
   - `attention_mask`: Mặt nạ nhị phân độ dài 30 (1 cho sự kiện thực tế, 0 cho vị trí padding).

### 2.2. Đầu ra & Hàm mất mát huấn luyện (Training Objectives)
1. **Trong quá trình huấn luyện (Pretraining Losses):**
   * **Masked Item / Field Reconstruction ($L_{\text{recon}}$):**
     * Che ngẫu nhiên 15% các token trong profile hoặc chuỗi hành vi bằng token `[MASK]`.
     * Sử dụng Cross-Entropy Loss cho các trường categorical (ngành hàng, loại hành vi) và MSE/Huber Loss cho các trường số học.
   * **Next-Action Prediction ($L_{\text{next}}$):**
     * Dựa trên chuỗi tiền tố (prefix sequence) trước cutoff, dự đoán hành vi hợp lệ đầu tiên xảy ra tiếp theo của người dùng.
   * **Tổng hàm mất mát kết hợp:**
     $$\mathcal{L}_{\text{total}} = \lambda_{\text{recon}} \mathcal{L}_{\text{recon}} + \lambda_{\text{next}} \mathcal{L}_{\text{next}}$$

2. **Khi suy luận (Inference Representation Extraction):**
   * Mô hình trích xuất ra **vector biểu diễn người dùng toàn diện**:
     * $z_{\text{long}}$: Vector biểu diễn dài hạn (Long-term preference) từ User ID Adapter và Tabular Profile.
     * $z_{\text{short}}$: Vector biểu diễn ngắn hạn (Short-term intent) thu được qua Attention Pooling trên các hidden states của chuỗi hành vi gần nhất.
     * $z_{\text{user}} = \text{Concat}[z_{\text{long}}, z_{\text{short}}]$ (hoặc qua một lớp MLP Fusion) làm đầu vào cho mọi bài toán Downstream.

---

## 3. DOWNSTREAM TASKS & KHẢ NĂNG ỨNG DỤNG THỰC TẾ

Vector biểu diễn $z_{\text{user}}$ sinh ra từ giai đoạn Pretraining có thể tái sử dụng trực tiếp hoặc Fine-tune cho 2 bài toán kinh điển:

### 3.1. Gợi ý sản phẩm / Dịch vụ (Food & E-commerce Recommendation)
* **Mục tiêu:** Dự đoán xác suất người dùng bấm vào xem ($P(\text{click})$) và đặt mua ($P(\text{purchase})$) một món ăn hoặc sản phẩm cụ thể.
* **Đầu vào:** Ghép vector người dùng $z_{\text{user}}$ với vector đặc trưng món ăn $v_{\text{item}}$ (giá tiền, nhóm ngành hàng, thương hiệu) và vector bối cảnh $v_{\text{context}}$ (khung giờ, địa điểm).
* **Đầu ra:** Điểm xếp hạng đề xuất $S = P(\text{click}) \times P(\text{purchase})$. Đánh giá qua thang đo **NDCG@K** và **HitRate@K**.

### 3.2. Tối ưu hóa quảng cáo hiển thị (Display Advertising CTR / CVR)
* **Mục tiêu:** Dự đoán tỷ lệ nhấp chuột (**CTR - Click-Through Rate**) và tỷ lệ chuyển đổi mua hàng (**CVR - Conversion Rate**).
* **Đầu vào:** Kết hợp $z_{\text{user}}$ với thuộc tính mẫu quảng cáo (`industry_main`, `ad_placement`, `creative_text`).
* **Đầu ra:** Điểm eCPM để tham gia cơ chế đấu giá thời gian thực (Real-time Bidding): $\text{eCPM} = \text{Bid} \times P(\text{CTR}) \times P(\text{CVR})$. Đánh giá qua **AUC-ROC** và **Logloss**.

---

## 4. HƯỚNG DẪN SỬ DỤNG & CÁC SCRIPT ĐỌC DỮ LIỆU

Thư mục `d:\Meituan\duc_data` đã được tích hợp đầy đủ các công cụ kiểm tra và nạp dữ liệu:

### 4.1. Cấu trúc thư mục dữ liệu & mã nguồn
```
duc_data/
├── README.md                      # Thẻ dữ liệu gốc từ Hugging Face
├── README_pretrain.md             # [TÀI LIỆU NÀY] Cấu trúc chi tiết bộ dữ liệu
├── pretraining_data_format.md     # Đặc tả chi tiết về format và các ràng buộc thời gian
├── preprocessing_config.json      # Cấu hình tiền xử lý, allowlist context, quy tắc chia split
├── manifest.json                  # Bảng kê sha256 checksums của 186 shard users
├── data_report.json               # Báo cáo thống kê toàn diện số liệu dataset
├── preview.json                   # Mẫu 10 snapshot tái lập đại diện
├── pretraining_dataset.py         # Module gốc dựng và duyệt dữ liệu
│
├── events/                        # 186 file Parquet (514 MB) - Mỗi file là toàn bộ event của 1 user
│   ├── user_001.parquet
│   └── ...
├── snapshots/                     # 186 file Parquet (36 MB) - Mỗi file là các điểm cắt snapshot của 1 user
│   ├── user_001.parquet
│   └── ...
│
├── read_data_basic.py             # Script đọc và in mẫu snapshot chi tiết theo split
├── inspect_parquet.py             # Script kiểm tra schema và đọc trực tiếp file Parquet
└── torch_dataset_loader.py        # Module PyTorch Dataset & DataLoader mẫu cho Pretraining
```

---

### 4.2. Hướng dẫn chạy các script

#### 🔹 Script 1: Đọc và duyệt mẫu dữ liệu snapshot (`read_data_basic.py`)
Đọc dữ liệu snapshot đã được tái cấu trúc hoàn chỉnh (gồm đầy đủ profile và chuỗi tối đa 30 events) theo logical split (`train`, `validation`, `test`):

```bash
# Xem 2 mẫu đầu tiên trong tập Train
python d:\Meituan\duc_data\read_data_basic.py --split train --num-samples 2

# Xem mẫu thuộc tập Test của riêng user_001
python d:\Meituan\duc_data\read_data_basic.py --split test --num-samples 1 --user-id user_001

# Xuất 5 mẫu ngẫu nhiên ra file JSON để phân tích
python d:\Meituan\duc_data\read_data_basic.py --num-samples 5 --export-json samples_demo.json
```

#### 🔹 Script 2: Kiểm tra trực tiếp file Parquet & Schema (`inspect_parquet.py`)
Đọc trực tiếp các tệp vật lý Parquet thông qua PyArrow cực nhanh, không tốn RAM:

```bash
# Xem thống kê tổng quan dung lượng, số file và schema cột
python d:\Meituan\duc_data\inspect_parquet.py --overview

# Xem 5 dòng đầu tiên trong bảng events của user_001
python d:\Meituan\duc_data\inspect_parquet.py --user user_001 --table events --limit 5

# Xem 3 dòng đầu tiên trong bảng snapshots của user_001
python d:\Meituan\duc_data\inspect_parquet.py --user user_001 --table snapshots --limit 3
```

#### 🔹 Script 3: PyTorch Streaming Dataset & DataLoader (`torch_dataset_loader.py`)
Tự động chuyển đổi profile số học thành vector và chuỗi event thành sequence tensor kèm attention mask:

```bash
# Chạy thử nghiệm DataLoader với batch size 4 trên tập Train
python d:\Meituan\duc_data\torch_dataset_loader.py --split train --batch-size 4 --limit 8
```

*Kết quả đầu ra của Batch:*
```
📦 [Batch #1]
   • profile_vector.shape : [4, 8]   -> Vector hồ sơ tĩnh 8 chiều
   • scenario_ids.shape   : [4, 30]  -> Chuỗi mã kịch bản độ dài 30
   • event_type_ids.shape : [4, 30]  -> Chuỗi loại hành động độ dài 30
   • attention_mask.shape : [4, 30]  -> Mặt nạ nhị phân xác định event thật vs padding
   • seq_lens trong batch  : [1, 2, 10, 12]
```

---

### 4.3. Code mẫu nhúng trực tiếp vào dự án Python
Nếu muốn viết code trực tiếp trong file của bạn:

```python
import sys
from pathlib import Path

# Trỏ đường dẫn tới thư mục duc_data
DATASET_DIR = Path("d:/Meituan/duc_data")
sys.path.insert(0, str(DATASET_DIR))

from pretraining_dataset import iter_pretraining_samples

# Lặp qua tập train
for sample in iter_pretraining_samples(DATASET_DIR, split="train"):
    sample_id = sample["sample_id"]
    user_id = sample["input"]["user_id"]
    cutoff = sample["input"]["cutoff_time"]
    profile = sample["input"]["profile"]
    history = sample["input"]["history"]  # Danh sách tối đa 30 events

    print(f"Sample: {sample_id} | User: {user_id} | Cutoff: {cutoff} | Events: {len(history)}")
    break
```
