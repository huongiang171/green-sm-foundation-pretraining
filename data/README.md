# Data

Không commit dataset thật vào Git.

## Expected data contract

Dữ liệu đầu vào tối thiểu nên có event-level records:

```text
user_id, timestamp, event_type, service, entity_id, value
```

Các trường nên có thêm khi dữ liệu cho phép: `session_id`, `time_gap`, `location`
và user profile features.

Ghi dataset version, nguồn lưu trữ và checksum trong experiment record hoặc data
catalog nội bộ.
