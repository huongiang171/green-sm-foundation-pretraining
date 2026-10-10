# Báo cáo dữ liệu pretraining User Foundation Model

## Mục đích và phạm vi

Tài liệu mô tả định dạng dữ liệu đầu vào cho giai đoạn pretraining của User Foundation Model trong PoC Hyper-personalization. Mỗi mẫu là một snapshot của một người dùng tại `cutoff_time`, gồm định danh người dùng, profile và lịch sử hành vi đã biết tại thời điểm đó. Mô hình sử dụng dữ liệu này để học representation có thể được đánh giá trên Food Recommendation và Advertising Optimization.

Các phần đầu mô tả đặc tả cơ sở. Bản dữ liệu OmniBehavior được triển khai theo `omnibehavior_pretraining_v2`, mô tả ở cuối tài liệu. Bản này ghi nhận thay đổi đơn vị event thành một action observation, giữ timestamp chỉ có ngày, timezone chưa rõ và thứ tự thời gian không xác định. Không áp dụng ngầm các giả định timestamp đầy đủ của đặc tả cơ sở cho bản v2.

Dataset lưu dữ liệu đầu vào gốc, không lưu `target`, `next_action_target`, `labels` hoặc dữ liệu đã bị mask trong từng record. Target cho Masked Reconstruction và Next-Action Prediction được tạo khi chuẩn bị dữ liệu huấn luyện. Việc bỏ target khỏi record không loại bỏ yêu cầu xác định nguồn target và kiểm soát thời gian.

Danh sách feature trong `profile` và thuộc tính bổ sung của từng event trong `history` có thể thay đổi theo dữ liệu thực tế. Các field trong ví dụ không phải bộ feature cố định cho mọi user hoặc mọi loại event.

Thiết kế tham khảo hướng kết hợp tabular profile, clickstream, `key:value` tokens và hai objective pretraining của [User foundation models for Grab](https://engineering.grab.com/user-foundation-models-for-grab). Cấu trúc JSON và các quy tắc dữ liệu dưới đây là đặc tả cho PoC này, không phải schema dữ liệu do Grab công bố.

## Một mẫu dữ liệu

Mẫu dưới đây là dữ liệu minh họa. Tại 11:50 ngày 01/09/2026, hệ thống đã ghi nhận ba event trong `history` và có profile tương ứng. Record chỉ chứa dữ liệu đầu vào, không chứa event tương lai hoặc target huấn luyện.

```json
{
  "sample_id": "U001_20260901_1150",
  "input": {
    "user_id": "U001",
    "cutoff_time": "2026-09-01T11:50:00+07:00",
    "profile": {
      "as_of_time": "2026-09-01T11:50:00+07:00",
      "city": "Hanoi",
      "avg_spend_30d": 85000,
      "preferred_service": "Food"
    },
    "history": [
      {
        "event_time": "2026-09-01T11:40:00+07:00",
        "event_type": "app_open",
        "service": "Food"
      },
      {
        "event_time": "2026-09-01T11:45:00+07:00",
        "event_type": "search",
        "service": "Food",
        "food_category": "noodles"
      },
      {
        "event_time": "2026-09-01T11:50:00+07:00",
        "event_type": "view",
        "service": "Food",
        "item_id": "F123"
      }
    ]
  }
}
```

`avg_spend_30d = 85000` trong ví dụ dùng đơn vị VND. Định nghĩa feature phải ghi rõ loại giao dịch được tính, phạm vi dịch vụ, công thức trung bình và cửa sổ 30 ngày kết thúc tại cutoff. `preferred_service` cũng cần có quy tắc xác định và cửa sổ thống kê; không mặc định là kết quả đếm mọi loại event.

## Cấu trúc của record

Các field cấu trúc cần được giữ ổn định để nối dữ liệu, kiểm tra thời gian và tạo training sample. Thuộc tính nghiệp vụ bên trong profile và event được phép mở rộng.

| Field | Kiểu dữ liệu | Yêu cầu và vai trò |
| --- | --- | --- |
| `sample_id` | String | Bắt buộc, duy nhất trong dataset; nhận diện snapshot, chống trùng mẫu và truy vết. Không phải feature của model. |
| `input` | Object | Bắt buộc; chứa dữ liệu người dùng của snapshot. |
| `input.user_id` | String | Bắt buộc; ổn định giữa các snapshot và các nguồn liên quan. Dùng cho User ID Adapter. |
| `input.cutoff_time` | Timestamp string | Bắt buộc; mốc giới hạn thông tin được phép xuất hiện trong input. Dùng kiểm tra thời gian và chia dữ liệu. |
| `input.profile` | Object | Bắt buộc; chứa metadata của profile và các feature user đã biết tại cutoff. |
| `input.profile.as_of_time` | Timestamp string | Bắt buộc trong format này; mốc hiệu lực của snapshot profile, không vượt `cutoff_time`. Không phải feature nghiệp vụ. |
| `input.history` | Array of objects | Bắt buộc; các event của cùng `user_id`, sắp theo thời gian. Encoder sử dụng tối đa 30 events gần nhất. |
| `input.history[].event_time` | Timestamp string | Bắt buộc cho mỗi event; dùng sắp xếp và tạo biểu diễn thời gian theo cấu hình. Không vượt `cutoff_time`. |
| `input.history[].event_type` | String | Bắt buộc cho mỗi event; loại hành vi như `app_open`, `search`, `view`, `click`, `booking`, `cancel` hoặc `complete`. |

User của từng event được xác định bởi `input.user_id`; không cần lặp lại `user_id` trong mọi event. Khi gom history từ nhiều nguồn, bước tạo dataset phải kiểm tra tất cả event thuộc đúng user.

Một mẫu có ba event vẫn hợp lệ; 30 events là giới hạn tối đa, không phải độ dài bắt buộc. Nếu nguồn chứa history dài hơn, preprocessing lấy 30 events gần nhất đủ điều kiện thời gian. Padding được tạo khi batching, không cần ghi thành event trong JSON. History rỗng phải được thống kê và chỉ sử dụng với các objective mà pipeline hỗ trợ; không tự tạo event để lấp chỗ trống.

## Feature có thể thay đổi và mở rộng

Các user feature bổ sung được đặt trong `profile`. Các thuộc tính hành vi bổ sung được đặt trong event tương ứng của `history`. Những event trong cùng một history không cần có tập field giống nhau.

| Nhóm | Field minh họa | Điều kiện sử dụng |
| --- | --- | --- |
| User categorical | `city`, `preferred_service` | Giá trị đã biết tại cutoff; mã hóa theo vocabulary của từng field. |
| User numerical | `avg_spend_30d`, `order_count_30d`, `ride_count_30d` | Ghi rõ đơn vị, công thức, loại tương tác và cửa sổ thống kê. |
| Event service | `service` | Phân biệt domain khi nguồn có nhiều dịch vụ; không giả định dataset có đủ Mobility, Food và Express. |
| Đối tượng Food | `item_id`, `restaurant_id`, `food_category` | Chỉ có ở event liên quan; ID phải nhất quán với nguồn catalog nếu dùng catalog. |
| Đối tượng quảng cáo | `ad_id`, `campaign_id`, `placement` | Chỉ có khi log thực tế cung cấp và thông tin đã biết tại event. |
| Event numerical | `fare`, `ETA`, `distance`, `amount` | Ghi rõ đơn vị và thời điểm giá trị sẵn có; không dùng kết quả cuối giao dịch cho event trước đó. |
| Location và context | `pickup_zone`, `dropoff_zone`, `device_type` | Chỉ sử dụng khi nguồn có field và đã định nghĩa cách mã hóa. |
| Metadata bổ sung | `event_id`, `session_id`, `available_at` | Hỗ trợ chống trùng, phân biệt thứ tự và kiểm tra thời điểm log sẵn có. Không tự động trở thành feature. |

Ví dụ, `app_open` có thể chỉ có thời gian, loại event và service; `search` có thể thêm `food_category`; `view` có thể thêm `item_id`. Một event Mobility có thể có location, fare hoặc ETA khi các field này thực sự tồn tại trong nguồn.

Khả năng thêm field vào JSON không đồng nghĩa checkpoint hiện tại tự động xử lý được field đó. Dataset cần đi kèm phiên bản schema và cấu hình feature, xác định tên field, kiểu dữ liệu, đơn vị, xử lý thiếu, adapter và việc field có được dùng làm reconstruction target hay không. Feature chưa được tokenizer hỗ trợ phải được báo cáo; không âm thầm đổi kiểu hoặc tạo embedding mới khi dùng checkpoint đã đóng băng.

Tên và ý nghĩa của field phải nhất quán giữa pretraining và downstream. Nếu hai nguồn dùng tên khác nhau, chẳng hạn `avg_spend_30d` và `avg_spend_30d_vnd`, preprocessing cần mapping đã ghi nhận để đưa về cùng feature. Không coi hai field là tương đương nếu công thức hoặc đơn vị khác nhau.

## Quy tắc thời gian và chất lượng

- Timestamp dùng ISO 8601 với timezone rõ ràng. Ví dụ dùng offset `+07:00`; khi so sánh timestamp từ nhiều nguồn, quy đổi về cùng chuẩn thời gian.
- Mọi event trong input phải đã xảy ra và đã sẵn có tại cutoff. `event_time <= cutoff_time` chưa đủ nếu log đến muộn; khi có `available_at`, kiểm tra cả thời điểm này.
- `as_of_time <= cutoff_time` là điều kiện metadata. Nội dung profile cũng phải được xây từ dữ liệu đã biết trước hoặc tại mốc hiệu lực đó. Chỉ sửa timestamp của profile không khắc phục leakage.
- Feature tổng hợp chỉ sử dụng dữ liệu quá khứ. Không tính `avg_spend_30d` hoặc `preferred_service` từ toàn bộ lịch sử gồm cả tương lai rồi gắn vào snapshot cũ.
- Event cùng timestamp cần quy tắc thứ tự có thể tái chạy, ưu tiên sequence number hoặc event ID từ nguồn. Nếu không xác định được thứ tự, không khẳng định quan hệ next-action giữa các event đó chỉ bằng cách sắp tên event.
- Loại log trùng dựa trên quy tắc đã định nghĩa. Các snapshot có history chồng lấn không mặc định là mẫu trùng; cách chọn cutoff và lấy mẫu cần được ghi nhận.
- Field vắng mặt hoặc `null` phải phân biệt với giá trị thực bằng `0`. Không tự suy ra các trường thiếu từ tên service hoặc loại event.
- Với các ID hoặc category chưa thấy khi train, áp dụng chính sách unknown đã lưu trong cấu hình. User mới không có embedding cá nhân đã học trong User ID Adapter; cần ghi nhận cách xử lý và độ phủ user.

## Tạo target trong quá trình huấn luyện

Training pipeline đọc record gốc, chọn các feature theo cấu hình rồi tạo input và target riêng cho từng objective. Metadata và target không được trộn vào feature của encoder.

### Masked Reconstruction

Pipeline chọn các tabular field hoặc event field đủ điều kiện, giữ giá trị gốc làm target trong bộ nhớ và thay phần được chọn bằng biểu diễn mask ở input huấn luyện. Record JSON gốc vẫn giữ nguyên.

Ví dụ với mẫu trên: chọn `profile.city`, thay token `city:Hanoi` bằng `city:[MASK]` và dùng `Hanoi` làm target categorical. Numerical feature có thể dùng MSE hoặc Huber Loss theo cấu hình; categorical feature dùng Cross-Entropy.

Mask policy phải ghi rõ tỷ lệ mask, đơn vị mask là field hay event, field được phép mask và cách xử lý các thuộc tính trùng lặp có thể tiết lộ đáp án. Chỉ tính reconstruction loss trên giá trị gốc thực sự quan sát được và đủ điều kiện; không coi field thiếu hoặc padding là đáp án cần tái tạo. Metadata như `sample_id`, `cutoff_time` và `as_of_time` không thuộc feature reconstruction trong format này.

### Next-Action Prediction

Target được tạo khi train, nhưng phải lấy từ hành vi có thật trong nguồn dữ liệu. Record minh họa dừng ở event `view` lúc 11:50; bản thân record không cho biết event sau 11:50 là gì. Không suy ra rằng user sẽ `order` chỉ vì trước đó đã `view`.

Có hai cách tạo mẫu next-action, tùy dữ liệu nguồn và cấu hình:

| Cách tạo | Input cho objective | Nguồn target và điều kiện |
| --- | --- | --- |
| Tra event tiếp theo từ log nguồn | Giữ profile và history tại cutoff của record. | Tra event hợp lệ đầu tiên sau cutoff của cùng user từ nguồn log. Event target chỉ được dùng tính loss, không đưa vào input. |
| Tách prefix từ history | Dùng các event trước một vị trí được chọn; đặt cutoff của prefix phù hợp. | Event tại vị trí được chọn làm target. Phải lấy hoặc tính lại profile đúng cutoff của prefix và loại target cùng mọi event sau target khỏi input. |

Với cách tách prefix, có thể dùng `app_open` lúc 11:40 và `search` lúc 11:45 để dự đoán `view` lúc 11:50. Tuy nhiên, profile có `as_of_time = 11:50` trong record gốc không mặc định hợp lệ cho input tại 11:45. Phải có snapshot profile tương ứng hoặc khả năng tính lại feature từ dữ liệu quá khứ. Nếu không bảo đảm điều kiện này, không sử dụng cách tách prefix đó.

Cấu hình phải định nghĩa action type được dự đoán và ý nghĩa của action value, chẳng hạn `item_id` khi event có đối tượng. Service có thể là thành phần của target theo cấu hình, nhưng không mặc định thay thế action value. Field target nào thiếu phải được loại khỏi loss tương ứng.

Event target phải nằm trong khoảng dữ liệu được phép dùng cho split huấn luyện và đã sẵn có trước cutoff huấn luyện. Không lấy event thuộc validation/test tương lai để tạo target cho train. Mẫu không có next event hợp lệ có thể chỉ tham gia Masked Reconstruction; không tự gán next-action giả hoặc coi việc thiếu log là một hành động của user.

Attention mask phải phù hợp với cách tạo objective. Next-action input chỉ chứa prefix trước target. Masked Reconstruction có thể dùng ngữ cảnh trong snapshot được phép, nhưng không đưa event tương lai ngoài cutoff vào input.

## Preprocessing và sử dụng trong mô hình

1. Kiểm tra cấu trúc record, ID, timestamp và điều kiện thời gian; làm sạch hoặc loại mẫu lỗi theo quy tắc đã ghi nhận.
2. Chuẩn hóa field theo schema, đơn vị và mapping của từng nguồn; phân biệt giá trị thiếu với số `0`.
3. Sắp history, xử lý trùng và lấy tối đa 30 events gần nhất tại cutoff.
4. Chuẩn bị input và target cho từng objective: chọn phần được mask hoặc prefix và next event. Kiểm tra lại profile/cutoff nếu tạo prefix; giữ target riêng với input.
5. Tạo các cặp `key:value` theo cấu hình và áp dụng mask cho input reconstruction. Ví dụ profile chưa mask có `city:Hanoi`; event `view` có `event_type:view`, `service:Food` và `item_id:F123`.
6. Tạo batch, position và attention mask; mã hóa ID/categorical bằng embedding, numerical bằng bucket embedding hoặc numeric projection. Căn chỉnh về `d_model = 128`, rồi đưa vào Shared Transformer Encoder gồm 2 layers và 4 attention heads. Timestamp phục vụ event position, time features hoặc time gap theo cấu hình, không mặc định là một category tùy ý.
7. Tính loss với target và loss mask tương ứng. Giữ ánh xạ giữa token, field và event để reconstruction và sequence aggregation dùng đúng vị trí.

Một event có thể tạo nhiều tokens nhưng vẫn chỉ tính là một event trong giới hạn 30. Các token của cùng event chia sẻ metadata thời gian/event phù hợp. Tabular features được xử lý như tập không có thứ tự; không biến thứ tự key trong JSON thành thứ tự hành vi.

`sample_id`, `cutoff_time` và `profile.as_of_time` phục vụ quản lý mẫu, kiểm tra thời gian và chia split, không đưa trực tiếp thành feature trong format này. `user_id` đi qua User ID Adapter. `event_time` dùng sắp xếp và tạo biểu diễn thời gian theo cấu hình.

`z_long` được trích trực tiếp từ User ID Adapter. `z_short` được aggregate từ hidden states của recent event tokens, loại padding và tabular tokens. Fusion để tạo `z_user` vẫn cần được chốt riêng; không suy ra phép cộng tensor hoặc kích thước `z_user` từ format dữ liệu.

## Lưu trữ và khả năng chạy lại

Có thể lưu các record cùng cấu trúc dưới dạng JSONL, mỗi dòng là một snapshot. Một user có thể có nhiều record tại các cutoff khác nhau. Dataset gốc giữ dữ liệu chưa mask và không có target; dữ liệu trung gian phục vụ training được tạo riêng khi cần.

Cấu hình đi kèm dataset và checkpoint cần lưu: nguồn dữ liệu, phiên bản schema/preprocessing, định nghĩa feature và đơn vị, quy tắc chọn snapshot, cutoff và mốc split, feature được dùng, vocabulary, bucket/normalization, mask policy, nguồn và cách tạo next-action target, random seed, các loss và trọng số `lambda_mask`, `lambda_next`.

Vocabulary, bucket boundaries và normalization phải fit trên phần train phù hợp. Chia dữ liệu theo thời gian, không random split; giữ validation/test để đánh giá theo protocol. Việc một user xuất hiện ở nhiều split không cho phép đưa hành vi tương lai của user đó vào pretraining hoặc profile train.

## Thông tin cần báo cáo khi có dataset thực tế

| Nhóm thống kê | Nội dung cần ghi nhận |
| --- | --- |
| Nguồn và thời gian | Nguồn log/profile, khoảng thời gian, timezone, độ trễ ghi nhận nếu có. |
| Quy mô | Số user, số snapshot, số event; số mẫu và user theo từng split. |
| Độ dài lịch sử | Phân bố số event/user và event/snapshot; tỷ lệ history rỗng, ngắn hoặc bị cắt về 30 events. |
| Độ phủ feature | Field thực tế ở profile và từng loại event; tỷ lệ thiếu, kiểu dữ liệu, vocabulary và tỷ lệ unknown. |
| Độ phủ hành vi | Phân bố event type, service và các đối tượng tương tác thực sự có trong dữ liệu. |
| Độ phủ objective | Số mẫu có reconstruction target hợp lệ; số mẫu có next-action target và action value hợp lệ; số mẫu bị bỏ qua và lý do. |
| Kiểm tra cutoff | Số trường hợp event/profile/target vi phạm thời gian, log trùng và quy tắc xử lý. |
| Liên kết downstream | Khả năng nối user và feature với dữ liệu Food/Advertising; không giả định hai task có sẵn log hoặc nhãn. |

Format này mô tả dữ liệu cần đưa vào pretraining và cho phép mở rộng feature. Nguồn dataset, số lượng mẫu và chất lượng thực tế cần được xác nhận bằng dữ liệu và thống kê; mẫu JSON minh họa không chứng minh dữ liệu đã sẵn sàng hoặc representation đã hữu ích cho downstream.

## Bản triển khai OmniBehavior: `omnibehavior_pretraining_v2`

Chương trình `scripts/build_pretraining_dataset.py` chuyển các file trong `raw_user_data/en` thành Parquet events và index snapshots. Hàm `iter_pretraining_samples(dataset_dir, split)` trong `scripts/pretraining_dataset.py` dựng record gồm `sample_id` và `input`; không thêm mask, target hoặc labels. `split` nhận `train`, `validation`, `test` hoặc `None` để đọc tất cả.

### Đơn vị event và bảo toàn dữ liệu

Mỗi phần tử của danh sách `action` trở thành một event. `event_type` giữ nguyên `action.type`; `scenario` giữ nguyên loại tương tác của raw record. Đây là loại observation, không tự động là hành động đã xảy ra thành công: `purchase` với `is_pay=false` vẫn giữ trạng thái âm. Không ánh xạ các scenario Kuaishou thành Food, Mobility hoặc Express.

`event_id = user_id:source_record_index:action_index`; index bắt đầu từ 0 trong file gốc. `source_record_id = user_id:source_record_index` liên kết các actions có chung context và timestamp. Hai raw records có nội dung giống nhau vẫn có ID khác nhau; không tự loại chúng như log trùng.

Parquet lưu `context_json` và `action_json` đầy đủ, giữ `false`, `0`, `null` và sự vắng mặt của key. Input có `action` là object gốc của một action và `context` đã lọc theo allowlist trong config. Context nguồn bằng `null` vẫn xuất hiện dưới dạng `null` trong input. Không dùng profile văn bản của nguồn.

### Timestamp, cutoff và thứ tự chưa rõ

Timezone nguồn chưa xác định; `input.timezone=null`. Timestamp đầy đủ chuyển thành `YYYY-MM-DDTHH:MM:SS`, không gắn offset. Timestamp chỉ ngày giữ `YYYY-MM-DD`. Đây là ngoại lệ được ghi nhận của yêu cầu timezone trong đặc tả cơ sở; không khẳng định các timestamp là UTC hoặc giờ Trung Quốc.

| Metadata của event | Ý nghĩa |
| --- | --- |
| `time_precision` | `second` hoặc `day`. |
| `time_lower_bound`, `time_upper_bound` | Với timestamp đầy đủ, hai mốc bằng nhau. Với ngày D, khoảng là đầu D đến đầu D+1. |
| `upper_bound_exclusive` | `true` cho khoảng ngày `[D, D+1)`; `false` cho timestamp đầy đủ. |
| `eligibility_time`, `eligibility_inclusive` | Mốc snapshot bắt đầu được phép chứa observation; không phải bằng chứng về thời điểm log sẵn có. |
| `source_record_id`, `action_index`, `source_action_count` | Liên kết actions cùng raw record và nhận diện nhóm bị cắt một phần. |

Event đủ giờ có eligibility coordinate `(timestamp, true)`. Event chỉ ngày D có coordinate `(đầu D+1, false)` và không xuất hiện trong snapshot sớm hơn lúc kết thúc D. Coordinate so sánh theo timestamp, rồi boolean: cutoff mở tại nửa đêm đứng trước cutoff đóng tại cùng timestamp. Vì vậy, snapshot đóng ngày trước không chứa event đúng 00:00 của ngày mới.

Snapshot tạo một lần sau mỗi eligibility coordinate. `input.cutoff_inclusive` phân biệt hai kiểu cutoff trên. History lấy tối đa 30 action-events theo eligibility coordinate, rồi source record index và action index. Đây là lựa chọn tái lập được theo mốc đủ điều kiện, không chứng minh 30 hành vi xảy ra gần nhất thực tế khi thời gian chỉ có ngày.

`input.history_order_known=false` khi các khoảng thời gian trong history chồng lấn hoặc có timestamp trùng. `history_truncated` cho biết prefix dài hơn 30 events. `history_group_truncated` cho biết biên 30 events cắt một nhóm cùng eligibility coordinate. Thứ tự trong array chỉ phục vụ serialization; không suy ra quan hệ next-action từ thứ tự action index hoặc thứ tự raw file. Training phải dùng khoảng thời gian và source record metadata để thiết kế position/attention phù hợp.

### Profile và time-based split

Profile có `as_of_time` bằng cutoff, `event_count_30d`, `active_days_30d` và sáu số đếm theo scenario. Cửa sổ profile là `(cutoff - 30 ngày, cutoff]` theo eligibility coordinates, gồm cả cờ inclusive của hai biên. Profile dùng toàn bộ observations đủ điều kiện trong cửa sổ, không chỉ history 30 events.

Các số đếm gồm trạng thái âm và không phải số giao dịch thành công. Ngày hoạt động lấy từ ngày lịch của timestamp nguồn. Một cửa sổ 30 ngày bắt đầu giữa ngày có thể chạm 31 ngày lịch. Các field là `advertisement_event_count_30d`, `live_streaming_event_count_30d`, `video_browsing_event_count_30d`, `e_commerce_event_count_30d`, `customer_service_event_count_30d`, `search_behavior_event_count_30d`.

Train bắt đầu 01/09/2025, kết thúc trước 01/11/2025; validation kết thúc trước 16/11/2025; test kết thúc trước 01/12/2025. Cutoff mở ở các biên thuộc split trước: snapshot đóng ngày 31/10 thuộc train, đóng ngày 15/11 thuộc validation, đóng ngày 30/11 thuộc test. Records 31/08 có thể cung cấp history nhưng không tạo snapshot ngoài khoảng split. Snapshot validation/test được phép chứa dữ liệu quá khứ của split trước.

### Context và mục tiêu training

Allowlist context được lưu đầy đủ trong `preprocessing_config.json`. Input dùng nội dung quảng cáo; tiêu đề, loại và nội dung livestream/video; mô tả, giá và danh mục sản phẩm; thông tin loại ticket/sản phẩm của chăm sóc khách hàng. Search Behavior không có context. Các thống kê tổng lượt xem, tương tác, doanh số 180 ngày và kết quả sau giao dịch bị loại khỏi context input nhưng vẫn giữ trong Parquet nguồn.

Các metadata ID, cutoff, eligibility và cờ kiểm tra không phải business features hoặc reconstruction targets. Thời gian và độ chính xác thời gian chỉ được dùng bởi cơ chế temporal encoding đã ghi nhận. Vocabulary, buckets và normalization phải fit từ các mẫu do reader trả về với `split="train"`; không fit từ toàn bộ bảng events vì bảng này giữ cả dữ liệu validation/test.

Khi tạo Masked Reconstruction, các bản sao context và field liên quan ở actions chung `source_record_id` cần được mask phối hợp để tránh lộ đáp án. Next-action target phải có thứ tự thời gian xác định, nằm trong khoảng training được phép và không có mặt trong input; không suy diễn cờ `false` thành hành động thành công. Chương trình hiện chỉ tạo dữ liệu đầu vào, chưa chốt mask policy, action-value target, losses hoặc nhãn CVR.

Nguồn không có `available_at`; việc lọc context và kiểm soát cutoff không chứng minh tuyệt đối rằng mọi giá trị đã sẵn có tại timestamp. Manifest, config và báo cáo giữ rõ giới hạn này.
