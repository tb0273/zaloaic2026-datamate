# Zalo AI Challenge 2026 — DataMate

Engine suy luận cho bài toán **DataMate**: đọc dữ liệu bán hàng thô của từng doanh nghiệp
(CSV / Excel / SQLite), sinh mã pandas để trả lời câu hỏi tiếng Việt, chạy mã trong sandbox
an toàn rồi ghi ra `submission.jsonl`.

Mô hình: `Qwen2.5-Coder-7B-Instruct` + LoRA adapter (fine-tuned), chạy 4-bit.

## Nội dung repo

| File | Mô tả |
|---|---|
| `datamate_inference_local.py` | Bản chạy local (Windows/Linux/macOS), GPU hoặc CPU |
| `datamate_inference_complete.ipynb` | Bản Google Colab (giữ để tham khảo) |
| `requirements-local.txt` | Thư viện cần cài |

**Không có trong repo** (tự tải về, xem bên dưới): `datamate_public_test/` (dữ liệu) và
`datamate_lora_adapter/` (trọng số LoRA, ~154MB).

## Cài đặt

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate

# torch bản CUDA trước (nếu có GPU NVIDIA), rồi mới cài phần còn lại
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements-local.txt
```

Kiểm tra GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Không có GPU vẫn chạy được bằng `--device cpu`, nhưng chậm hơn hàng chục lần.

## Chuẩn bị dữ liệu

Đặt 2 thư mục cạnh script:

```
datamate_inference_local.py
datamate_public_test/       # questions.jsonl, calendar.csv, data/dev_001..dev_014
datamate_lora_adapter/
    lora_adapter/           # adapter_config.json, adapter_model.safetensors, tokenizer...
```

Tải trước base model (~5GB) để thấy tiến trình và resume được nếu ngắt:

```powershell
python datamate_inference_local.py --prefetch
```

## Chạy

```powershell
# Kiểm tra dữ liệu + prompt, không cần model (nhanh)
python datamate_inference_local.py --dry-run --limit 3

# Chạy thử vài câu, in mã sinh ra để soi
python datamate_inference_local.py --limit 5 --verbose 5 --show-errors

# Chạy full 1000 câu (tự resume nếu Ctrl+C, tự lưu stats)
python datamate_inference_local.py --output submission.jsonl --log-file run.log

# Chạy lại chỉ các câu đang abstain
python datamate_inference_local.py --output submission.jsonl --retry-abstain

# Kiểm tra format trước khi nộp
python datamate_inference_local.py --verify-only --output submission.jsonl
```

Tuỳ chọn hữu ích: `--ids`, `--business`, `--offset`, `--overwrite`,
`--max-prompt-tokens`, `--max-new-tokens`, `--sandbox-timeout`, `--max-attempts`,
`--no-literal-guard`, `--no-type-hints`. Xem đầy đủ bằng `--help`.

## Cách hoạt động

**1. Nạp dữ liệu** (`load_biz_tables`)
- Đọc **một lần** mỗi file Excel bằng `calamine` (Rust), fallback `openpyxl`.
- Tự dò dòng tiêu đề (header lệch dòng 1/2/3) và **bỏ các dòng tiêu đề lặp lại** giữa file.
- Chuẩn hoá số kiểu Việt Nam: `79k` → 79000, `69.000đ` → 69000, `12,4%` → 12.4.
- Chuẩn hoá ngày: serial Excel `46023` → `2026-01-01`, `1 thg 3 2025`, `dd/mm/yyyy`,
  giữ `HH:MM:SS` khi có, và `1/1` (thiếu năm) lấy năm từ tên file.

**2. Chọn bảng đưa vào prompt**
- Chấm điểm theo từ khoá câu hỏi và theo **tháng/năm trong tên bảng**.
- Câu hỏi không nêu tháng ("hôm nay", "tuần này") thì dùng tháng/năm của `as_of_date`.
- Schema tự co giãn để vừa ngân sách token.

**3. Prompt**
- System prompt chứa luật nghiệp vụ then chốt: loại đơn huỷ/trả (giữ đơn đang giao),
  doanh thu = tổng thành tiền dòng hàng, kỳ tính theo ngày đặt, cấm hardcode tên,
  cấm dùng `now()`.
- Dòng **`Kỳ cần tính: YYYY-MM-DD → YYYY-MM-DD`** được tính sẵn từ `as_of_date`.

**4. Sandbox** (`execute_sandbox`)
- Timeout bằng thread (hoạt động trên Windows, khác `signal.alarm`).
- `pd` là proxy: `pd.read_csv('don_hang.csv')` trả về bảng đã làm sạch.
- Shim `DataFrame.append` / `applymap` (pandas 3 đã bỏ) để code kiểu cũ vẫn chạy.
- **Đồng hồ đóng băng**: `pd.Timestamp.now()`, `datetime.now()`, `pd.to_datetime('today')`
  đều trả `as_of_date`.

**5. Chống đáp án bịa**
- Nếu code chứa list tên không tồn tại trong dữ liệu → sinh lại (guard
  `find_hallucinated_literals`); hết lượt thì `abstain` thay vì đoán bừa.
- Câu hỏi một đáp án mà model trả list → lấy phần tử đầu.
- Kết quả `NaN`/rỗng → `abstain` (0 điểm) thay vì đoán (−1 điểm).

**6. Ghi kết quả**
- Ghi ngay từng dòng, có resume; cuối run sắp xếp lại theo thứ tự `questions.jsonl`
  và tự kiểm tra format (`--verify-only`).

## Hiệu năng đo được

| Việc | Trước | Sau |
|---|---|---|
| Nạp 14 doanh nghiệp (1.4M dòng) | 264s | **43s** |
| Sinh 1 câu (RTX 3060, 4-bit) | — | ~20s (~12.6 tok/s) |
| Sinh 1 câu (CPU) | ~5.2 phút | — |

1000 câu trên RTX 3060 khoảng 5–6 giờ. Nếu GPU đang bị game/ứng dụng khác chiếm VRAM,
script sẽ cảnh báo trước khi nạp model.

## Ghi chú

- Windows không cài được Unsloth → script tự fallback `transformers + peft` (vẫn chạy GPU).
- Dữ liệu cuộc thi và trọng số mô hình không được đưa vào repo này.
