#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataMate 2026 - Local Inference Runner
======================================

Ban local (Windows / Linux / macOS) cua notebook `datamate_inference_complete.ipynb`
(truoc day chi chay duoc tren Google Colab).

Thay doi so voi ban Colab:
  * Khong con `google.colab` / Google Drive / `!pip install` / upload file.
  * Khong con `signal.SIGALRM` (khong ton tai tren Windows) -> timeout bang thread.
  * Nap model theo 3 lop fallback: unsloth -> transformers + peft (bnb 4-bit) ->
    transformers + peft (fp16/bf16/CPU).
  * Duong dan tuong doi theo thu muc chua script, ho tro tham so dong lenh.
  * Resume / checkpoint / ghi file theo dung thu tu cau hoi / thong ke cuoi run.

Vi du:
    python datamate_inference_local.py --dry-run --limit 3
    python datamate_inference_local.py --limit 20 --business dev_010
    python datamate_inference_local.py --output submission_local.jsonl
    python datamate_inference_local.py --verify-only --output submission_local.jsonl

Yeu cau (xem requirements-local.txt):
    pip install torch transformers peft accelerate bitsandbytes pandas numpy openpyxl
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import datetime as dt
import io
import json
import math
import os
import re
import sqlite3
import sys
import threading
import time
import traceback
import warnings
from collections import OrderedDict
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Log sach: bo qua canh bao phien ban cua pandas/transformers
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

# --------------------------------------------------------------------------- #
# 0. Duong dan mac dinh (tuong doi theo vi tri script)
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = HERE / "datamate_public_test"
DEFAULT_ADAPTER = HERE / "datamate_lora_adapter" / "lora_adapter"
DEFAULT_OUTPUT = HERE / "submission_local.jsonl"

SYSTEM_PROMPT = (
    "Bạn là trợ lý phân tích dữ liệu DataMate. Chỉ trả về một khối mã Python duy nhất "
    "trong ```python ... ``` để tính toán kết quả và gán vào biến `result`. "
    "Tuyệt đối không thêm văn bản giải thích.\n\n"
    "Luật bắt buộc khi tính toán:\n"
    "- Doanh thu / số đơn / số khách / xếp hạng: CHỈ loại đơn có trạng thái Huỷ và Trả hàng; "
    "GIỮ đơn Hoàn thành VÀ đơn Đang giao (đơn trả hàng dù chỉ 1 phần thì bỏ cả đơn).\n"
    "- Doanh thu = tổng thành tiền các dòng hàng (đã trừ giảm giá từng dòng); "
    "KHÔNG trừ giảm giá cả đơn, KHÔNG cộng VAT, KHÔNG gồm phí giao hàng.\n"
    "- Đơn thuộc kỳ theo NGÀY ĐẶT, không theo ngày giao hay ngày thu tiền. "
    "Phải lọc theo đúng kỳ ở dòng 'Kỳ cần tính' bên dưới.\n"
    "- Ngày đã ở dạng chuỗi 'YYYY-MM-DD' (có thể kèm ' HH:MM:SS'): so sánh/nhóm bằng "
    "pd.to_datetime(...) hoặc .str[:10]. Lọc theo tháng thì dùng .str[:7] == 'YYYY-MM', "
    "KHÔNG so sánh cột ngày với chuỗi 'YYYY-MM'.\n"
    "- Câu hỏi về tiền mặt/ghi nợ dùng cột hình thức thanh toán ghi trên đơn, "
    "không dùng các lần trả nợ về sau.\n"
    "- Bản ghi có ngày SAU as_of_date coi như chưa xảy ra (trừ trạng thái huỷ/trả của đơn).\n"
    "- 'Hôm nay' chính là as_of_date. TUYỆT ĐỐI KHÔNG dùng pd.Timestamp.now(), "
    "datetime.now(), datetime.today() hay pd.to_datetime('today') để tính kỳ.\n"
    "- Câu hỏi hỏi MỘT thứ (nào, ai, ngày nào, giờ nào, bao nhiêu) thì result là MỘT giá trị, "
    "không trả list. Chỉ trả list khi câu hỏi hỏi nhiều giá trị (liệt kê, top N, các tháng...).\n"
    "- Chỉ dùng đúng tên bảng/cột có trong schema, không tự bịa tên cột.\n"
    "- KHÔNG được hardcode danh sách tên sản phẩm / khách / nhân viên / nhà cung cấp. "
    "Phải lọc từ dữ liệu: .str.contains(...), isin(df['cột'].unique()), hoặc groupby. "
    "Ví dụ sai: tea = ['Trà chanh (L)', 'Trà xanh (L)']; df[df['Tên'].isin(tea)].\n"
    "- Nếu dữ liệu không đủ để trả lời: gán result = '__UNANSWERABLE__'.\n"
    "- Nếu không chắc: gán result = '__ABSTAIN__' thay vì đoán."
)

ABSTAIN_SENTINELS = {
    "__ABSTAIN__", "#VALUE!", "NaN", "nan", "None", "", "NaT",
    "-", "—", "n/a", "N/A", "null", "NULL", "không có dữ liệu",
}
UNANSWERABLE_SENTINEL = "__UNANSWERABLE__"

_LOG_LOCK = threading.Lock()
_LOG_PATH: Optional[Path] = None


def _setup_console() -> None:
    """Bat UTF-8 cho console Windows (tranh loi UnicodeEncodeError voi tieng Viet)."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def log(msg: str = "") -> None:
    with _LOG_LOCK:
        print(msg, flush=True)
        if _LOG_PATH is not None:
            try:
                with _LOG_PATH.open("a", encoding="utf-8") as fh:
                    fh.write(msg + "\n")
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# 1. Tien ich thoi gian
# --------------------------------------------------------------------------- #
def parse_as_of(as_of_date: str) -> dt.date:
    try:
        return dt.datetime.strptime(str(as_of_date).strip(), "%Y-%m-%d").date()
    except Exception:
        return dt.date.today()


def resolve_time(query: str, as_of_date: str) -> str:
    """Mo ta tuong doi cua moc thoi gian trong cau hoi (giong ban Colab)."""
    curr_dt = parse_as_of(as_of_date)
    q = query.lower()
    if "hôm nay" in q:
        return f"hôm nay ({curr_dt.isoformat()})"
    if "hôm qua" in q:
        return f"hôm qua ({(curr_dt - dt.timedelta(days=1)).isoformat()})"
    if "tháng này" in q:
        return f"tháng này ({curr_dt.month}/{curr_dt.year})"
    if "tháng trước" in q:
        m = 12 if curr_dt.month == 1 else curr_dt.month - 1
        y = curr_dt.year - 1 if curr_dt.month == 1 else curr_dt.year
        return f"tháng trước ({m}/{y})"
    return f"ngày neo ({as_of_date})"


def resolve_period(query: str, as_of_date: str) -> Optional[Tuple[dt.date, dt.date]]:
    """Doi cach noi tuong doi (hom nay, tuan nay, thang truoc...) thanh khoang ngay cu the.

    Rulebook: moi cach noi tuong doi tinh theo as_of_date, khong theo ngay that.
    """
    q = query.lower()
    anchor = parse_as_of(as_of_date)

    def month_range(year: int, month: int) -> Tuple[dt.date, dt.date]:
        first = dt.date(year, month, 1)
        next_month = dt.date(year + (month == 12), (month % 12) + 1, 1)
        return first, next_month - dt.timedelta(days=1)

    if "hôm nay" in q or "hom nay" in q:
        return anchor, anchor
    if "hôm qua" in q or "hom qua" in q:
        day = anchor - dt.timedelta(days=1)
        return day, day
    if "tuần này" in q or "tuan nay" in q:
        return anchor - dt.timedelta(days=anchor.weekday()), anchor
    if "tuần trước" in q or "tuần rồi" in q or "tuan truoc" in q:
        start = anchor - dt.timedelta(days=anchor.weekday() + 7)
        return start, start + dt.timedelta(days=6)
    if "tháng này" in q or "thang nay" in q:
        return anchor.replace(day=1), anchor
    if "tháng trước" in q or "thang truoc" in q:
        first_this = anchor.replace(day=1)
        last_prev = first_this - dt.timedelta(days=1)
        return last_prev.replace(day=1), last_prev
    if "từ đầu năm" in q or "đầu năm nay" in q:
        return dt.date(anchor.year, 1, 1), anchor

    match = re.search(r"(\d{1,3})\s*ng[àa]y\s*(?:gần nhất|gan nhat|trở lại|tro lai|qua)", q)
    if match:
        days = max(1, int(match.group(1)))
        return anchor - dt.timedelta(days=days - 1), anchor

    match = re.search(r"th[áa]ng\s*(\d{1,2})(?:\s*[/\-]\s*(\d{4})|\s*n[ăa]m\s*(\d{4}))?", q)
    if match:
        month = int(match.group(1))
        year_txt = match.group(2) or match.group(3)
        if 1 <= month <= 12:
            if year_txt:
                return month_range(int(year_txt), month)
            # Khong ghi nam: nam nay neu khong sau hom nay, nguoc lai nam truoc
            first, last = month_range(anchor.year, month)
            return (first, last) if first <= anchor else month_range(anchor.year - 1, month)

    return None


# --------------------------------------------------------------------------- #
# 2. Chuan hoa so / ngay kieu Viet Nam
# --------------------------------------------------------------------------- #
_UNIT_MULTIPLIERS = (
    (("nghìn", "nghin", "ngàn", "ngan"), 1_000.0),
    (("triệu", "trieu", "tr"), 1_000_000.0),
    (("tỷ", "ty", "tỉ"), 1_000_000_000.0),
)

_NUM_RE = re.compile(r"^([+-]?[\d.,\s]+?)\s*([^\d\s.,]{0,12})$")


def parse_vn_number(value: Any) -> float:
    """`79k` -> 79000, `69.000đ` -> 69000, `1,5 lít` -> 1.5, `12%` -> 12.0, `5 kg` -> 5.0."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "#value!", "#n/a", "nan%", "-", "—"}:
        return float("nan")

    text = text.replace("\u00a0", " ").replace("₫", "").replace("đ", "").strip()
    match = _NUM_RE.match(text)
    if not match:
        return float("nan")

    num_str, unit = match.group(1).replace(" ", ""), match.group(2).strip()
    if not num_str:
        return float("nan")

    # 1.234.567 / 1,234,567 / 1.234,56 / 12,5
    if "," in num_str and "." in num_str:
        if num_str.rfind(",") > num_str.rfind("."):
            num_str = num_str.replace(".", "").replace(",", ".")
        else:
            num_str = num_str.replace(",", "")
    elif "." in num_str:
        num_str = re.sub(r"\.(\d{3})(?!\d)", r"\1", num_str)
    elif "," in num_str:
        num_str = re.sub(r",(\d{3})(?!\d)", r"\1", num_str).replace(",", ".")

    try:
        parsed = float(num_str)
    except Exception:
        return float("nan")

    # Chi nhan don vi viet tat dinh luong, KHONG nham voi "kg", "m", "ly", "cây"...
    if unit == "k" and re.search(r"\d\s*k$", text):
        parsed *= 1_000.0
    else:
        for units, factor in _UNIT_MULTIPLIERS:
            if unit in units:
                parsed *= factor
                break
    return parsed


def _looks_like_quantity_column(col_name: str) -> bool:
    """Ten cot mang tien / so luong (dung de ha nguong nhan dien so)."""
    keys = (
        "số lượng", "so luong", "quy đổi", "quy doi", "đơn giá", "don gia", "giá bán",
        "gia ban", "giá nhập", "gia nhap", "giá vốn", "gia von", "thành tiền", "thanh tien",
        "doanh thu", "tổng tiền", "tong tien", "số tiền", "so tien", "còn nợ", "con no",
        "tiền giảm", "tien giam", "chi phí", "chi phi", "lãi", "lai", "tồn kho", "ton kho",
        "số lượng nhập", "số lượng bán", "cộng", "cong", "trả trước", "tra truoc",
    )
    low = col_name.lower()
    return any(k in low for k in keys)


def _map_cached(series: pd.Series, func) -> pd.Series:
    """`series.map(func)` co cache theo gia tri (du lieu lap lai rat nhieu -> nhanh hon nhieu)."""
    memo: Dict[Any, Any] = {}

    def apply(value: Any) -> Any:
        try:
            if value in memo:
                return memo[value]
        except TypeError:
            return func(value)
        result = func(value)
        memo[value] = result
        return result

    return series.map(apply)


def _downcast_integral(series: pd.Series) -> pd.Series:
    """Cot so nguyen (khong NaN) -> int64 cho de doc."""
    try:
        if series.notna().all() and (series % 1 == 0).all():
            return series.astype("int64")
    except Exception:
        pass
    return series


def clean_numeric_series(series: pd.Series, col_name: str = "") -> pd.Series:
    """Chuyen cot chuoi co don vi sang so, an toan voi cot khong phai so."""
    if not isinstance(series, pd.Series) or series.empty:
        return series
    if is_date_column(col_name, series):
        return series
    if series.dtype.kind in ("i", "f", "b"):
        return series

    # Fast path: cot von da la so thuan (so sanh vector hoa, nhanh hon map rat nhieu)
    fast = pd.to_numeric(series, errors="coerce")
    valid = series.notna().sum()
    if valid and fast.notna().sum() / valid >= 0.95:
        return _downcast_integral(fast)

    sample = series.dropna().astype(str).head(30)
    if sample.empty:
        return series

    converted = [parse_vn_number(v) for v in sample]
    numeric_ratio = sum(1 for v in converted if not math.isnan(v)) / len(sample)

    should_convert = numeric_ratio >= 0.6 or (
        _looks_like_quantity_column(col_name) and numeric_ratio >= 0.3
    )
    if should_convert:
        numeric = pd.to_numeric(_map_cached(series, parse_vn_number), errors="coerce")
        return _downcast_integral(numeric)
    return series


_DATE_NAME_KEYS = (
    "ngày", "ngay", "tháng", "thang", "date", "thời gian", "thoi gian",
    "giờ", "gio", "lúc", "luc", "hạn", "sinh nhật", "sinh nhat",
    "đăng ký", "dang ky", "hết hạn", "het han", "time",
)

_MONEY_NAME_KEYS = (
    "tiền", "tien", "giá", "gia", "doanh thu", "thành tiền", "thanh tien",
    "tổng", "tong", "nợ", "chi phí", "chi phi", "lãi", "lai", "vốn", "von",
    "phí", "phi", "giảm", "giam", "toán", "toan", "số lượng", "so luong",
    "sl", "cộng", "cong", "trả", "tra", "lương", "luong", "điểm", "diem",
)

_DATE_LIKE_RE = re.compile(
    r"^\s*(\d{1,4}[/-]\d{1,2}[/-]\d{1,4}|\d{1,2}\s*thg\s*\d{1,2}\s*\d{4}|"
    r"\d{4}-\d{2}-\d{2}[ T]\d{1,2}:\d{2}|[3-5]\d{4}(\.0)?)\s*$"
)


def _name_has(name_low: str, keys: Iterable[str]) -> bool:
    """Khop ten cot voi tu khoa; key ASCII ngan phai khop ca tu (tranh 'han' khop 'thanh')."""
    for key in keys:
        if key not in name_low:
            continue
        if len(key) <= 3 and key.isascii():
            if re.search(rf"(?<!\w){re.escape(key)}(?!\w)", name_low):
                return True
            continue
        return True
    return False


def is_date_column(col_name: str, series: Optional[pd.Series] = None) -> bool:
    """Cot ngay/thoi gian? Uu tien ten cot, sau do doan theo noi dung."""
    low = str(col_name).lower()

    # Ten noi ro la ngay/gio -> luon la cot ngay (ke ca "Ngay thanh toan")
    strong = ("ngày", "ngay", "date", "thời gian", "thoi gian", "giờ", "gio",
              "hạn", "sinh nhật", "sinh nhat", "tháng", "thang", "time")
    if any(k in low for k in strong):
        return True

    # Ten mang tien/so luong -> KHONG phai cot ngay (vi du "Tổng thanh toán")
    if _name_has(low, _MONEY_NAME_KEYS):
        return False

    # Cot da la so -> khong the la cot ngay (vi du "Giá vốn lúc bán")
    if isinstance(series, pd.Series) and series.dtype.kind in ("i", "f"):
        return False

    # Doan theo noi dung (cot ten la, gia tri la ngay)
    if isinstance(series, pd.Series) and series.dtype.kind == "O":
        sample = series.dropna().astype(str).head(12)
        if len(sample) and sum(1 for v in sample if _DATE_LIKE_RE.match(v)) / len(sample) >= 0.6:
            return True
    return False


def _format_timestamp(ts: pd.Timestamp) -> str:
    """Giu ca gio:phut:giay khi co (can cho cau hoi 'gio cao diem')."""
    if ts.hour or ts.minute or ts.second:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    return ts.strftime("%Y-%m-%d")


_VN_THG_RE = re.compile(r"^(\d{1,2})\s*thg\s*(\d{1,2})\s*,?\s*(\d{4})$")


def _parse_datetime_value(val: Any) -> Any:
    """Chuan hoa 1 gia tri ngay: Excel serial, '1 thg 3 2025', dd/mm/yyyy, ISO..."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    if isinstance(val, (dt.datetime, dt.date, pd.Timestamp)):
        return _format_timestamp(pd.Timestamp(val))

    text = str(val).strip()
    if not text or text.lower() in {"nan", "none", "nat", "#value!", "#n/a", "-", "—"}:
        return None

    # Serial ngay cua Excel (46023 -> 2026-01-01)
    try:
        as_float = float(text)
        if 20000 < as_float < 80000:
            return _format_timestamp(pd.to_datetime(as_float, unit="D", origin="1899-12-30"))
    except Exception:
        pass

    # Dinh dang tieng Viet: "1 thg 3 2025"
    match = _VN_THG_RE.match(text)
    if match:
        day, month, year = (int(x) for x in match.groups())
        try:
            return _format_timestamp(pd.Timestamp(year=year, month=month, day=day))
        except Exception:
            return text

    dayfirst = not re.match(r"^\d{4}-\d{1,2}-\d{1,2}", text)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(text, dayfirst=dayfirst, errors="coerce")
        if pd.notna(parsed):
            return _format_timestamp(pd.Timestamp(parsed))
    except Exception:
        pass
    return text


_NO_FFILL_KEYS = ("đăng ký", "dang ky", "sinh nhật", "sinh nhat", "hạn dùng", "han dung")

# Dinh dang ngay hay gap trong du lieu. Parse bang C cua pandas (nhanh gap ~100 lan
# so voi de dateutil doan tung o).
_DATE_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%Y/%m/%d",
    "%d/%m/%y",
)

_VN_THG_EXTRACT = re.compile(r"^\s*(\d{1,2})\s*thg\s*(\d{1,2})\s*,?\s*(\d{4})\s*$")


def _to_ns(parsed: Any) -> pd.Series:
    """Chuan hoa ve datetime64[ns] va loai gia tri ngoai pham vi (nam 1, nam 200...)."""
    out = pd.to_datetime(parsed, errors="coerce")
    index = getattr(out, "index", None)
    if not isinstance(out, pd.Series) or out.dtype.kind != "M":
        return pd.Series(pd.NaT, index=index, dtype="datetime64[ns]")
    try:
        lo, hi = pd.Timestamp("1900-01-01"), pd.Timestamp("2100-12-31")
        out = out.mask((out < lo) | (out > hi))
    except Exception:
        pass
    try:
        return out.astype("datetime64[ns]")
    except Exception:
        return pd.Series(pd.NaT, index=index, dtype="datetime64[ns]")


def _year_from_name(*names: str) -> Optional[int]:
    """Nam ghi trong ten file/sheet (rulebook: ngay khong ghi nam lay nam cua sheet)."""
    for name in names:
        if not name:
            continue
        match = re.search(r"(20\d{2})", str(name))
        if match:
            return int(match.group(1))
    return None


def _vectorized_datetime(
    series: pd.Series,
    default_year: Optional[int] = None,
) -> Tuple[pd.Series, pd.Series]:
    """Parse ca cot ngay bang vector hoa. Tra ve (datetime Series, mask chua parse duoc)."""
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    no_pending = pd.Series(False, index=series.index)

    if series.dtype.kind == "M":
        return _to_ns(series), no_pending

    values = series.astype(object)
    pending = values.notna()

    # Gia tri da la datetime/date that (calamine hay tra ve kieu nay)
    dt_mask = values.map(
        lambda v: isinstance(v, (dt.datetime, dt.date, pd.Timestamp)) if pd.notna(v) else False
    )
    if bool(dt_mask.any()):
        converted = _to_ns(values[dt_mask])
        result = result.fillna(converted.reindex(values.index))
        pending = result.isna() & values.notna()

    for fmt in _DATE_FORMATS:
        if not bool(pending.any()):
            break
        try:
            parsed = _to_ns(pd.to_datetime(values.where(pending), format=fmt, errors="coerce"))
        except Exception:
            continue
        result = result.fillna(parsed)
        pending = result.isna() & values.notna()

    # "1 thg 3 2025"
    if bool(pending.any()):
        sub = values.where(pending).astype(str).str.extract(
            r"^\s*(\d{1,2})\s*thg\s*(\d{1,2})\s*,?\s*(\d{4})\s*$"
        )
        ok = sub.notna().all(axis=1)
        if bool(ok.any()):
            rebuilt = (
                sub.loc[ok, 2].astype(int).astype(str)
                + "-" + sub.loc[ok, 1].astype(int).astype(str).str.zfill(2)
                + "-" + sub.loc[ok, 0].astype(int).astype(str).str.zfill(2)
            )
            parsed = _to_ns(pd.to_datetime(rebuilt, format="%Y-%m-%d", errors="coerce"))
            result = result.fillna(parsed.reindex(values.index))
            pending = result.isna() & values.notna()

    # "1/1" (ngay khong ghi nam) -> lay nam cua file/sheet
    if bool(pending.any()) and default_year:
        sub = values.where(pending).astype(str).str.extract(r"^\s*(\d{1,2})\s*[/-]\s*(\d{1,2})\s*$")
        ok = sub.notna().all(axis=1)
        if bool(ok.any()):
            rebuilt = (
                str(int(default_year))
                + "-" + sub.loc[ok, 1].astype(int).astype(str).str.zfill(2)
                + "-" + sub.loc[ok, 0].astype(int).astype(str).str.zfill(2)
            )
            parsed = _to_ns(pd.to_datetime(rebuilt, format="%Y-%m-%d", errors="coerce"))
            result = result.fillna(parsed.reindex(values.index))
            pending = result.isna() & values.notna()

    # Serial ngay cua Excel (46023 -> 2026-01-01)
    if bool(pending.any()):
        nums = pd.to_numeric(values.where(pending), errors="coerce")
        serial_mask = nums.gt(20000) & nums.lt(80000)
        if bool(serial_mask.any()):
            serials = _to_ns(
                pd.to_datetime(nums.where(serial_mask), unit="D", origin="1899-12-30",
                               errors="coerce")
            )
            result = result.fillna(serials)
            pending = result.isna() & values.notna()

    return result, pending


def clean_date_series(
    series: pd.Series,
    col_name: str = "",
    default_year: Optional[int] = None,
) -> pd.Series:
    """Chuan hoa cot ngay ve ISO; giu gio:phut:giay neu co; ffill cho o gop nhom."""
    if not isinstance(series, pd.Series) or series.empty:
        return series

    parsed, unresolved = _vectorized_datetime(series, default_year=default_year)

    # Phan con lai (dinh dang la): chi parse tren cac gia tri DUY NHAT
    if bool(unresolved.any()):
        leftovers = series[unresolved].astype(str)
        memo = {value: _parse_datetime_value(value) for value in leftovers.unique()}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fixed = _to_ns(pd.to_datetime(leftovers.map(memo), errors="coerce"))
        parsed = parsed.copy()
        parsed.loc[unresolved] = fixed.values

    if bool(parsed.notna().any()):
        # NaT.hour la NaN nen phai fillna(0), neu khong cot ngay se bi them 00:00:00
        hours = parsed.dt.hour.fillna(0)
        minutes = parsed.dt.minute.fillna(0)
        seconds = parsed.dt.second.fillna(0)
        has_time = bool(((hours != 0) | (minutes != 0) | (seconds != 0)).any())
        fmt = "%Y-%m-%d %H:%M:%S" if has_time else "%Y-%m-%d"
        cleaned = parsed.dt.strftime(fmt).astype(object)
    else:
        cleaned = pd.Series([None] * len(series), index=series.index, dtype=object)

    # Gia tri khong hieu duoc -> giu nguyen text goc (khong xoa du lieu)
    if bool(unresolved.any()):
        cleaned.loc[unresolved] = series[unresolved].astype(str).values
    cleaned = cleaned.where(series.notna(), None)

    # O trong o cac dong sau cua cung nhom = gia tri dong dau nhom.
    # Nhung cot cap thuc the (ngay dang ky, sinh nhat...) thi o trong la thieu that.
    low = str(col_name).lower()
    if not any(k in low for k in _NO_FFILL_KEYS):
        cleaned = cleaned.ffill()
    return cleaned


# --------------------------------------------------------------------------- #
# 3. Doc du lieu doanh nghiep (tu dong do header)
# --------------------------------------------------------------------------- #
HEADER_KEYWORDS = (
    "ngày", "ngay", "mã", "ma", "tên", "ten", "stt", "tiền", "tien", "sl",
    "số lượng", "so luong", "giá", "gia", "hàng", "hang", "khách", "khach",
    "đvt", "dvt", "trạng thái", "trang thai", "loại", "loai", "imei", "sđt",
    "nợ", "no", "đg", "chi", "nhập", "nhap", "doanh thu", "bán", "ban",
    "đơn", "don", "size", "màu", "mau", "nhóm", "nhom", "kho", "lô", "lo",
)


def auto_detect_header(df_raw: pd.DataFrame) -> int:
    """Tim dong tieu de thuc su khi header nam o dong 1 / 2 / 3."""
    best_row, best_score = 0, -1.0
    limit = min(8, len(df_raw))
    for row_idx in range(limit):
        row_vals = [str(v).lower().strip() for v in df_raw.iloc[row_idx] if pd.notna(v)]
        if not row_vals:
            continue
        kw_matches = sum(1 for v in row_vals if any(k in v for k in HEADER_KEYWORDS))
        text_count = sum(1 for v in row_vals if not re.fullmatch(r"[\d.,\-\s]*", v))
        score = kw_matches * 4 + text_count
        if score > best_score and (kw_matches >= 2 or (kw_matches >= 1 and text_count >= 3)):
            best_row, best_score = row_idx, score
    return best_row


def _drop_repeated_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Bo cac dong lap lai tieu de (file Excel noi nhieu bang, moi block co header rieng).

    Neu khong bo, cac dong nay lam sai so don / so khach / so mat hang.
    """
    if df.empty or len(df.columns) < 2:
        return df
    first = df.columns[0]
    try:
        first_norm = df[first].astype(str).str.strip().str.lower()
        first_is_header = first_norm.eq(str(first).strip().lower())
    except Exception:
        return df
    if not bool(first_is_header.any()):
        return df

    names = {str(c).strip().lower() for c in df.columns}
    matches = df.astype(str).apply(lambda s: s.str.strip().str.lower().isin(names)).sum(axis=1)
    header_rows = first_is_header & matches.ge(2)
    if bool(header_rows.any()):
        return df.loc[~header_rows].reset_index(drop=True)
    return df


def _finalize_frame(df: pd.DataFrame, default_year: Optional[int] = None) -> pd.DataFrame:
    """Chuan hoa ten cot + kieu du lieu cho moi bang."""
    if df is None or df.empty:
        return df

    seen: Dict[str, int] = {}
    columns: List[str] = []
    for col in df.columns:
        name = str(col).strip()
        if not name or name.lower().startswith("unnamed"):
            name = f"col_{len(columns)}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        columns.append(name)
    df.columns = columns

    # Bo cot rong hoan toan
    df = df.dropna(axis=1, how="all")
    # Bo dong rong hoan toan
    df = df.dropna(axis=0, how="all").reset_index(drop=True)
    # Bo dong tieu de lap lai giua file
    df = _drop_repeated_header_rows(df)
    if df.empty:
        return df

    for col in df.columns:
        series = df[col]
        try:
            if is_date_column(col, series):
                df[col] = clean_date_series(series, col_name=col, default_year=default_year)
            elif series.dtype.kind == "O":
                df[col] = clean_numeric_series(series, col_name=col)
        except Exception as exc:
            # Mot cot loi khong duoc lam mat ca bang
            log(f"   ⚠️  Cột '{col}' giữ nguyên dữ liệu thô ({type(exc).__name__}: {exc})")
    return df


def _read_csv_any(path: Path, header: Optional[int]) -> pd.DataFrame:
    last_error: Optional[Exception] = None
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, header=header, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc
            break
    if last_error:
        raise last_error
    return pd.DataFrame()


_EXCEL_ENGINE: Optional[str] = None
_EXCEL_ENGINE_CHECKED = False


def _excel_engine() -> Optional[str]:
    """Engine doc Excel nhanh nhat co san: calamine (Rust) > openpyxl (Python thuan)."""
    global _EXCEL_ENGINE, _EXCEL_ENGINE_CHECKED
    if not _EXCEL_ENGINE_CHECKED:
        _EXCEL_ENGINE_CHECKED = True
        try:
            import python_calamine  # noqa: F401

            _EXCEL_ENGINE = "calamine"
        except Exception:
            _EXCEL_ENGINE = None
    return _EXCEL_ENGINE


def _read_excel_sheets(path: Path) -> Dict[str, pd.DataFrame]:
    """Doc TAT CA sheet trong 1 lan (openpyxl rat cham neu doc lap lai tung sheet)."""
    engine = _excel_engine()
    kwargs: Dict[str, Any] = dict(sheet_name=None, header=None)
    if engine:
        kwargs["engine"] = engine
    try:
        sheets = pd.read_excel(path, **kwargs)
    except Exception:
        # Engine nhanh khong xu ly duoc file -> quay ve mac dinh
        kwargs.pop("engine", None)
        sheets = pd.read_excel(path, **kwargs)

    if not isinstance(sheets, dict):
        return {path.stem: sheets}
    return {str(name): frame for name, frame in sheets.items()}


def load_biz_tables(biz_folder: Path) -> Dict[str, pd.DataFrame]:
    """Doc toan bo file cua mot doanh nghiep thanh dict {ten_bang: DataFrame}."""
    tables: Dict[str, pd.DataFrame] = {}
    if not biz_folder.exists():
        return tables

    for path in sorted(p for p in biz_folder.iterdir() if p.is_file()):
        if path.suffix.lower() in {".md", ".txt", ".json", ".zip"}:
            continue
        stem = re.sub(r"\W|^(?=\d)", "_", path.stem)
        file_year = _year_from_name(path.stem)
        try:
            suffix = path.suffix.lower()
            if suffix == ".csv":
                raw = _read_csv_any(path, header=None)
                if raw.empty:
                    continue
                header_row = auto_detect_header(raw.head(8))
                df = _read_csv_any(path, header=header_row)
                tables[stem] = _finalize_frame(df, default_year=file_year)

            elif suffix in (".xlsx", ".xls", ".xlsm", ".xltx"):
                sheets = _read_excel_sheets(path)
                for sheet, raw in sheets.items():
                    if raw is None or raw.empty:
                        continue
                    header_row = auto_detect_header(raw.head(8))
                    df = raw.iloc[header_row + 1:].copy().reset_index(drop=True)
                    df.columns = [str(v).strip() for v in raw.iloc[header_row].values]
                    sheet_key = re.sub(r"\W|^(?=\d)", "_", str(sheet))
                    key = f"{stem}_{sheet_key}" if len(sheets) > 1 else stem
                    tables[key] = _finalize_frame(
                        df, default_year=_year_from_name(sheet) or file_year
                    )

            elif suffix in (".sqlite", ".db", ".sqlite3"):
                with sqlite3.connect(path) as conn:
                    names = pd.read_sql_query(
                        "SELECT name FROM sqlite_master WHERE type='table';", conn
                    )["name"].tolist()
                    for name in names:
                        safe_table = re.sub(r"\W|^(?=\d)", "_", str(name))
                        key = f"{stem}_{safe_table}"
                        df = pd.read_sql_query(f'SELECT * FROM "{name}"', conn)
                        tables[key] = _finalize_frame(df, default_year=file_year)
        except Exception as exc:
            log(f"   ⚠️  Bỏ qua {path.name}: {exc}")
    return tables


class TableCache:
    """Cache bang da doc theo doanh nghiep (tranh doc lai 1000 lan).

    max_items=0 -> giu tat ca trong RAM (khuyen nghi: cau hoi bi xao tron giua cac
    doanh nghiep nen cache nho se lam doc lai file Excel lien tuc).
    """

    def __init__(self, root: Path, max_items: int = 0):
        self.root = root
        self.max_items = max_items
        self._cache: "OrderedDict[str, Dict[str, pd.DataFrame]]" = OrderedDict()

    def get(self, biz_id: str) -> Dict[str, pd.DataFrame]:
        if biz_id in self._cache:
            self._cache.move_to_end(biz_id)
            return self._cache[biz_id]
        tables = load_biz_tables(self.root / biz_id)
        self._cache[biz_id] = tables
        while self.max_items and len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return tables


def load_calendar(data_root: Path) -> Optional[pd.DataFrame]:
    path = data_root / "calendar.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        return df
    except Exception as exc:
        log(f"⚠️  Không đọc được calendar.csv: {exc}")
        return None


# --------------------------------------------------------------------------- #
# 4. Schema gon cho prompt
# --------------------------------------------------------------------------- #
def build_compact_schema(
    tables: Dict[str, pd.DataFrame],
    sample_rows: int = 2,
    max_cols: int = 12,
) -> str:
    lines: List[str] = []
    for name, df in tables.items():
        if df is None or not isinstance(df, pd.DataFrame):
            continue
        cols = [str(c) for c in list(df.columns)[:max_cols]]
        line = f"- Bảng `{name}` ({len(df)} dòng):\n  Cột: {cols}"
        if sample_rows > 0 and not df.empty:
            sample = df.head(sample_rows).to_dict(orient="records")
            line += f"\n  Mẫu: {json.dumps(sample, ensure_ascii=False, default=str)}"
        lines.append(line)
    return "\n".join(lines)


FACT_TABLE_HINTS = ("don", "order", "hoa_don", "hoadon", "ban", "sale", "chitiet", "chi_tiet", "nhap", "phieu")


def _query_time_hints(query: str) -> Tuple[set, set]:
    """Nam / thang duoc nhac trong cau hoi (de chon dung bang theo thang)."""
    low = query.lower()
    years = {int(y) for y in re.findall(r"\b(20\d{2})\b", low)}
    months = set()
    for raw in re.findall(r"th[áa]ng\s*(\d{1,2})", low):
        value = int(raw)
        if 1 <= value <= 12:
            months.add(value)
    for raw in re.findall(r"\bt(\d{1,2})(?!\d)", low):
        value = int(raw)
        if 1 <= value <= 12:
            months.add(value)
    return years, months


def _name_time_hints(table_name: str) -> Tuple[set, set]:
    """Nam / thang ma ten bang the hien, vi du `BanHang_T11_2025` -> ({2025}, {11})."""
    low = table_name.lower()
    years = {int(y) for y in re.findall(r"20\d{2}", low)}
    months = set()
    for raw in re.findall(r"(?:t|thang|th[áa]ng|m)(\d{1,2})(?!\d)", low):
        value = int(raw)
        if 1 <= value <= 12:
            months.add(value)
    return years, months


def _word_score(word: str, haystack: str) -> float:
    if len(word) >= 4:
        return 1.0 if word in haystack else 0.0
    return 1.0 if re.search(rf"(?<!\w){re.escape(word)}(?!\w)", haystack) else 0.0


def _is_comparison_question(query: str) -> bool:
    return any(
        key in query.lower()
        for key in ("so với", "so voi", "trước", "truoc", "chênh", "chenh",
                    "tăng", "tang", "giảm", "giam", "quý", "quy")
    )


def time_hints_for(query: str, as_of_date: Optional[str]) -> Tuple[set, set]:
    """Thang/nam de chon bang: lay tu cau hoi, neu khong co thi lay tu as_of_date.

    Vi du cau "doanh thu tuần này" khong neu thang -> phai dùng tháng của as_of_date,
    neu khong se chon nham bang thang khác và lọc ra 0 dòng.
    """
    years, months = _query_time_hints(query)
    if (years or months) or not as_of_date:
        return years, months

    anchor = parse_as_of(as_of_date)
    years, months = {anchor.year}, {anchor.month}
    if _is_comparison_question(query):
        prev_month = 12 if anchor.month == 1 else anchor.month - 1
        prev_year = anchor.year - 1 if anchor.month == 1 else anchor.year
        months = {anchor.month, prev_month}
        years = {anchor.year, prev_year}
    return years, months


def score_tables(
    query: str,
    tables: Dict[str, pd.DataFrame],
    years: Optional[set] = None,
    months: Optional[set] = None,
) -> List[Tuple[float, str]]:
    """Diem khop giua tu khoa cau hoi va ten bang / ten cot (uu tien bang phat sinh)."""
    q_words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 1]
    if years is None or months is None:
        q_years, q_months = _query_time_hints(query)
        years = q_years if years is None else years
        months = q_months if months is None else months
    q_years, q_months = years, months
    scored: List[Tuple[float, str]] = []
    for name, df in tables.items():
        name_low = re.sub(r"[_\-]", " ", str(name).lower())
        score = 0.0
        for word in q_words:
            if _word_score(word, name_low):
                score += 3.0
            for col in df.columns:
                if _word_score(word, str(col).lower()):
                    score += 1.0
        if any(hint in name_low for hint in FACT_TABLE_HINTS):
            score += 1.5

        # Uu tien bang dung thang / nam duoc hoi (du lieu thuong tach sheet theo thang)
        t_years, t_months = _name_time_hints(str(name))
        if q_years and t_years:
            score += 12.0 if (t_years & q_years) else -6.0
        if q_months and t_months:
            score += 10.0 if (t_months & q_months) else -4.0

        scored.append((score, name))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored


def get_top_tables(
    query: str,
    tables: Dict[str, pd.DataFrame],
    top_k: int = 2,
) -> Dict[str, pd.DataFrame]:
    if top_k <= 0 or not tables or len(tables) <= top_k:
        return tables
    ranked = [name for _, name in score_tables(query, tables)[:top_k]]
    return {name: tables[name] for name in ranked}


def build_schema_within_budget(
    query: str,
    tables: Dict[str, pd.DataFrame],
    calendar: Optional[pd.DataFrame],
    max_chars: int,
    top_k: int,
    sample_rows: int,
    as_of_date: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Chon so bang / so dong mau sao cho schema vua ngan sach token cua prompt."""
    q_years, q_months = time_hints_for(query, as_of_date)
    ranked = score_tables(query, tables, q_years, q_months)
    default_table = ranked[0][1] if ranked else None
    order = [name for _, name in ranked]

    # Bo cac bang chac chan thuoc thang/nam khac voi cau hoi (vi du hoi thang 11 -> bo T9)
    # Tru khi cau hoi co so sanh (so voi thang truoc, cung ky nam truoc...) thi can giu lai.
    comparison = _is_comparison_question(query)
    if (q_years or q_months) and not comparison and len(order) > 1:
        consistent = []
        for name in order:
            t_years, t_months = _name_time_hints(name)
            wrong_month = q_months and t_months and not (t_months & q_months)
            wrong_year = q_years and t_years and not (t_years & q_years)
            if wrong_month or wrong_year:
                continue
            consistent.append(name)
        if consistent:
            order = consistent

    if top_k and top_k > 0 and not (q_years and not q_months):
        # Cau hoi theo nam (khong neu thang) can du cac bang thang -> khong cat bot
        order = order[:top_k]
    if not order:
        order = list(tables)

    calendar_line = ""
    if calendar is not None:
        calendar_line = (
            f"\n- Bảng `calendar` ({len(calendar)} dòng): lịch âm, "
            "cột [date, weekday, lunar_day, lunar_month, lunar_year, lunar_leap, holiday]"
        )

    budget = max_chars - len(calendar_line) - len(query) - 450
    smallest = ""

    # 1) Uu tien giu du bang, giam so dong mau truoc
    for rows in range(sample_rows, -1, -1):
        subset = {name: tables[name] for name in order}
        candidate = build_compact_schema(subset, sample_rows=rows)
        smallest = candidate
        if len(candidate) <= budget:
            return candidate + calendar_line, default_table

    # 2) Van qua dai -> bot dan bang it lien quan nhat
    for count in range(len(order) - 1, 0, -1):
        subset = {name: tables[name] for name in order[:count]}
        candidate = build_compact_schema(subset, sample_rows=0)
        smallest = candidate
        if len(candidate) <= budget:
            return candidate + calendar_line, default_table

    return smallest + calendar_line, default_table


# --------------------------------------------------------------------------- #
# 5. Sandbox (timeout hoat dong tren Windows)
# --------------------------------------------------------------------------- #
class SandboxTimeout(Exception):
    pass


def _install_pandas_shims() -> None:
    """Model hay sinh code kieu pandas 1.x; pandas 3 da bo .append/.applymap.

    Them shim de code cu van chay thay vi loi AttributeError roi phai chay lai.
    """
    if not hasattr(pd.DataFrame, "append"):
        def _append(self, other, ignore_index=False, verify_integrity=False, sort=False):
            frames = [self] + (list(other) if isinstance(other, (list, tuple)) else [other])
            return pd.concat(frames, ignore_index=ignore_index, sort=sort)

        pd.DataFrame.append = _append  # type: ignore[attr-defined]

    if not hasattr(pd.DataFrame, "applymap"):
        pd.DataFrame.applymap = lambda self, func, *a, **k: self.map(func, *a, **k)  # type: ignore[attr-defined]

    if not hasattr(pd.Series, "append"):
        def _sappend(self, other, ignore_index=False, verify_integrity=False):
            parts = [self] + (list(other) if isinstance(other, (list, tuple)) else [other])
            return pd.concat(parts, ignore_index=ignore_index)

        pd.Series.append = _sappend  # type: ignore[attr-defined]


_install_pandas_shims()


class SafePandas:
    """`pd` trong sandbox: read_csv('don_hang.csv') tu dong tra ve bang da lam sach."""

    def __init__(self, real_pd: Any, tables: Dict[str, pd.DataFrame], as_of_date: Optional[str] = None):
        self._pd = real_pd
        self._tables = tables
        self._as_of = parse_as_of(as_of_date) if as_of_date else None
        self._timestamp_cls = None
        if self._as_of is not None:
            as_of, real = self._as_of, real_pd

            class _FrozenTimestamp(real.Timestamp):  # type: ignore[misc]
                """Timestamp.now() tra ve as_of_date, khong phai ngay that cua may."""

                @classmethod
                def now(cls, tz=None):
                    return real.Timestamp(as_of)

                @classmethod
                def today(cls, tz=None):
                    return real.Timestamp(as_of)

                @classmethod
                def utcnow(cls):
                    return real.Timestamp(as_of)

            self._timestamp_cls = _FrozenTimestamp

    def __getattr__(self, item: str) -> Any:
        if item == "Timestamp" and self._timestamp_cls is not None:
            return self._timestamp_cls
        return getattr(self._pd, item)

    def _lookup(self, filepath: Any) -> Optional[pd.DataFrame]:
        stem = Path(str(filepath)).stem
        if stem in self._tables:
            return self._tables[stem]
        low = stem.lower()
        for key, value in self._tables.items():
            if low in key.lower() or key.lower() in low:
                return value
        return None

    def read_csv(self, filepath, *args, **kwargs):
        found = self._lookup(filepath)
        return found if found is not None else self._pd.read_csv(filepath, *args, **kwargs)

    def read_excel(self, filepath, *args, **kwargs):
        found = self._lookup(filepath)
        return found if found is not None else self._pd.read_excel(filepath, *args, **kwargs)

    def sqlite3_connect(self, *args, **kwargs):
        return sqlite3.connect(*args, **kwargs)

    def to_datetime(self, arg, *args, **kwargs):
        # 'today'/'now' phai la as_of_date, khong phai ngay that
        if self._as_of is not None and isinstance(arg, str) and arg.strip().lower() in ("today", "now"):
            return self._pd.Timestamp(self._as_of)
        try:
            return self._pd.to_datetime(arg, *args, **kwargs)
        except Exception:
            kwargs.pop("format", None)
            kwargs["dayfirst"] = True
            kwargs["errors"] = "coerce"
            return self._pd.to_datetime(arg, *args, **kwargs)


def _make_time_shims(as_of_date: str) -> Any:
    """`datetime` trong sandbox: now()/today() tra ve as_of_date."""
    as_of = parse_as_of(as_of_date)

    class _FrozenDatetime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(as_of.year, as_of.month, as_of.day)

        @classmethod
        def today(cls):
            return cls(as_of.year, as_of.month, as_of.day)

        @classmethod
        def utcnow(cls):
            return cls(as_of.year, as_of.month, as_of.day)

    class _FrozenDate(dt.date):
        @classmethod
        def today(cls):
            return cls(as_of.year, as_of.month, as_of.day)

    return SimpleNamespace(
        datetime=_FrozenDatetime,
        date=_FrozenDate,
        time=dt.time,
        timedelta=dt.timedelta,
        timezone=dt.timezone,
        MINYEAR=dt.MINYEAR,
        MAXYEAR=dt.MAXYEAR,
        __name__="datetime",
    )


_FORBIDDEN_CODE = re.compile(
    r"\b(shutil\.rmtree|os\.remove|os\.unlink|os\.rmdir|subprocess\.|os\.system)\b"
)


_LANG_TAGS = {"python", "py", "python3", "pandas", "ipython"}


def _split_fences(text: str) -> List[str]:
    """Tach moi khoi ```...``` (ke ca khoi bi cat cut, khong co ``` dong)."""
    blocks: List[str] = []
    pos = 0
    while pos < len(text):
        start = text.find("```", pos)
        if start == -1:
            break
        newline = text.find("\n", start)
        if newline == -1:
            # "```python result = 1" tren cung dong
            body = text[start + 3:]
            tag, _, first = body.partition(" ")
            body = first if tag.strip().lower() in _LANG_TAGS else body
            blocks.append(body.strip())
            break
        body = text[newline + 1:]
        end = body.find("```")
        if end == -1:
            blocks.append(body.strip())  # bi cat token: lay het phan con lai
            break
        blocks.append(body[:end].strip())
        pos = newline + 1 + end + 3
    return blocks


def extract_code(raw: str) -> str:
    """Lay khoi code Python; chiu duoc ca truong hop bi cat token (thieu ``` dong)."""
    if not raw:
        return ""
    text = raw.replace("<|im_end|>", "").replace("<|im_start|>", "")

    blocks = [b for b in _split_fences(text) if b]
    if blocks:
        # Uu tien khoi co gan ket qua, trong do lay khoi CUOI (thuong la ban sua)
        with_result = [b for b in blocks if re.search(r"\b(result|res|answer|ans)\s*=", b)]
        code = (with_result or blocks)[-1]
    else:
        code = text

    # Bo dong dau chi ghi ten ngon ngu ("python", "py"...) do bi cat cut
    lines = code.split("\n")
    while lines and lines[0].strip().strip("`").lower() in _LANG_TAGS:
        lines.pop(0)
    result = "\n".join(lines).strip()
    # Bo ``` dong con sot lai (truong hop "```python result = 3```")
    return re.sub(r"```\s*$", "", result).strip()


def _clean_generated_code(code: str) -> str:
    lines = []
    for line in code.split("\n"):
        stripped = line.strip()
        if stripped.startswith(("Dưới đây", "Để trả lời", "Here is", "Note:", "Giải thích")):
            continue
        if stripped.startswith(("import sys", "sys.exit", "quit(", "exit(")):
            continue
        # `import pandas as pd` se ghi de SafePandas -> pd.read_csv that bai
        if re.fullmatch(
            r"\s*(import\s+(pandas|numpy)(\s+as\s+\w+)?|from\s+(pandas|numpy)\s+import\s+\w+)\s*",
            line,
        ):
            continue
        lines.append(line)
    return "\n".join(lines)


def _unique_values_of(
    tables: Dict[str, pd.DataFrame],
    default_table: Optional[str],
    max_unique: int = 50000,
) -> set:
    """Tập giá trị chuỗi có thực trong các bảng, để phát hiện giá trị bịa (có cache)."""
    cache_key = id(tables)
    cached = _LITERAL_VALUE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    values: set = set()
    for df in tables.values():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        for col in df.columns:
            series = df[col]
            if series.dtype.kind != "O":
                continue
            try:
                uniq = series.dropna().unique()
            except Exception:
                continue
            if len(uniq) <= max_unique:
                values.update(str(v).strip().lower() for v in uniq)

    if len(_LITERAL_VALUE_CACHE) > 4:
        _LITERAL_VALUE_CACHE.clear()
    _LITERAL_VALUE_CACHE[cache_key] = values
    return values


_LITERAL_VALUE_CACHE: Dict[int, set] = {}

# Tu khoa nghiep vu luon hop le (rulebook), khong duoc coi la "bia"
_ALLOWED_LITERALS = {
    "đã huỷ", "đã hủy", "huỷ", "hủy", "đã hoàn thành", "hoàn thành", "đang giao",
    "đã hoàn hàng", "trả hàng", "hoàn hàng", "xong", "cash", "transfer", "ewallet",
    "cod", "debt", "tiền mặt", "chuyển khoản", "ví điện tử", "ghi nợ",
    "giao hàng thu tiền", "nhập đầu kỳ", "bán", "nhập", "hủy đơn", "đã trả hàng",
}

_LIST_LITERAL_RE = re.compile(r"\[([^\[\]]*?)\]", re.DOTALL)
_STR_LITERAL_RE = re.compile(r"['\"]([^'\"]{2,})['\"]")


def find_hallucinated_literals(
    code: str,
    tables: Dict[str, pd.DataFrame],
    default_table: Optional[str] = None,
) -> List[str]:
    """Chuỗi hardcode trong list mà KHÔNG có trong dữ liệu.

    Model hay bịa danh sách sản phẩm (vd 'Trà xanh (L)') rồi lọc theo đó -> đáp án sai.
    """
    literals: List[str] = []
    for match in _LIST_LITERAL_RE.finditer(code):
        found = _STR_LITERAL_RE.findall(match.group(1))
        # 1 phan tu = truy cap cot df['Tên sản phẩm'], khong phai danh sach gia tri
        if len(found) >= 2:
            literals.extend(found)
    if not literals:
        return []

    known = _unique_values_of(tables, default_table)
    if not known:
        return []

    missing: List[str] = []
    for lit in literals:
        norm = lit.strip().lower()
        if norm in _ALLOWED_LITERALS or len(norm) < 5:
            continue
        if norm not in known:
            missing.append(lit)
    return missing


def _call_with_timeout(func, timeout_s: float):
    box: Dict[str, Any] = {}

    def runner():
        try:
            box["value"] = func()
        except BaseException as exc:  # noqa: BLE001 - can bat moi thu tu sandbox
            box["error"] = exc

    thread = threading.Thread(target=runner, name="datamate-sandbox", daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        HUNG_SANDBOX_THREADS.append(thread)
        raise SandboxTimeout(f"Code chạy quá {timeout_s:g}s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


HUNG_SANDBOX_THREADS: List[threading.Thread] = []


class _NullWriter:
    """Ghi ra hu khong (dung cho --dry-run de khong tao file submission rong)."""

    def write(self, *_args, **_kwargs) -> int:
        return 0

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def execute_sandbox(
    code_str: str,
    tables: Dict[str, pd.DataFrame],
    as_of_date: str,
    timeout_s: float = 8.0,
    default_table: Optional[str] = None,
) -> Tuple[bool, Any, str]:
    """Tra ve (ok, value, error_message)."""
    code = _clean_generated_code(extract_code(code_str))
    if not code:
        return False, None, "empty_code"
    if _FORBIDDEN_CODE.search(code):
        return False, None, "forbidden_code"

    globs: Dict[str, Any] = {
        "pd": SafePandas(pd, tables, as_of_date),
        "np": np,
        "math": math,
        "re": re,
        "datetime": _make_time_shims(as_of_date),
        "sqlite3": sqlite3,
        "tables": tables,
        "as_of_date": as_of_date,
        "today": parse_as_of(as_of_date),
        "result": None,
        "res": None,
        "answer": None,
        "ans": None,
        "output": None,
        "final_result": None,
    }
    if tables:
        # `df` = bang lien quan nhat (theo diem khop), khong phai bang dau tien
        first_key = default_table if default_table in tables else next(iter(tables))
        globs["df"] = tables[first_key]
        for name, frame in tables.items():
            globs[name] = frame
            globs[re.sub(r"[^a-zA-Z0-9_]", "_", name)] = frame

    stdout_buf = io.StringIO()

    def run() -> Any:
        try:
            parsed = ast.parse(code)
        except SyntaxError:
            parsed = None

        if parsed is not None and parsed.body and isinstance(parsed.body[-1], ast.Expr):
            prefix = ast.Module(body=parsed.body[:-1], type_ignores=[])
            with contextlib.redirect_stdout(stdout_buf):
                exec(compile(prefix, "<sandbox>", "exec"), globs)
                value = eval(
                    compile(ast.Expression(parsed.body[-1].value), "<sandbox>", "eval"), globs
                )
            if value is not None:
                return value

        with contextlib.redirect_stdout(stdout_buf):
            exec(compile(code, "<sandbox>", "exec"), globs)

        for name in ("result", "res", "answer", "ans", "output", "final_result"):
            if globs.get(name) is not None:
                return globs[name]
        printed = stdout_buf.getvalue().strip()
        if printed:
            return printed.split("\n")[-1].strip()
        return None

    saved_stdout = sys.stdout
    try:
        value = _call_with_timeout(run, timeout_s)
    except SandboxTimeout as exc:
        return False, None, str(exc)
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"
    finally:
        # Neu thread bi bo roi giua redirect_stdout, phai tra stdout ve de khong mat log
        if sys.stdout is not saved_stdout:
            sys.stdout = saved_stdout

    if value is None:
        printed = stdout_buf.getvalue().strip()
        return False, None, f"no_result (stdout={printed[:120]!r})"
    return True, value, ""


# --------------------------------------------------------------------------- #
# 6. Chuan hoa dap an theo rulebook
# --------------------------------------------------------------------------- #
def _round_number(value: float) -> Any:
    if math.isnan(value) or math.isinf(value):
        return None
    if math.isclose(value, round(value), abs_tol=1e-6):
        return int(round(value))
    return round(float(value), 2)


def infer_type(value: Any) -> Tuple[str, Any]:
    """Chuyen gia tri Python -> (type, answer) dung format rulebook."""
    if value is None:
        return "abstain", None

    if isinstance(value, pd.DataFrame):
        if value.empty:
            return "abstain", None
        if value.shape == (1, 1):
            value = value.iloc[0, 0]
        elif value.shape[1] == 1:
            value = value.iloc[:, 0].tolist()
        elif value.shape[0] == 1:
            value = value.iloc[0, :].tolist()
        else:
            return "abstain", None

    if isinstance(value, pd.Series):
        clean = value.dropna()
        if len(clean) == 0:
            return "abstain", None
        value = clean.iloc[0] if len(clean) == 1 else clean.tolist()

    # DatetimeIndex / Index khac (df['ngay'].unique(), groupby(...).apply())
    if isinstance(value, pd.Index):
        value = value.tolist()

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, np.datetime64):
        value = pd.Timestamp(value)

    if isinstance(value, (bool, np.bool_)):
        return "boolean", bool(value)

    if isinstance(value, (dt.datetime, dt.date, pd.Timestamp)):
        return "date", pd.Timestamp(value).strftime("%Y-%m-%d")

    if isinstance(value, Decimal):
        value = float(value)

    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        rounded = _round_number(float(value))
        return ("abstain", None) if rounded is None else ("number", rounded)

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return "abstain", None
        # 1 phan tu -> coi la gia tri don (giong Series/Timestamp), tranh sai format list
        if len(value) == 1:
            return infer_type(value[0])
        items = []
        for item in value:
            _, converted = infer_type(item)
            items.append(converted)
        return "list", items

    if isinstance(value, dict):
        if "columns" in value:  # bang
            return "table", value
        return "abstain", None

    text = str(value).strip()
    if text in ABSTAIN_SENTINELS:
        return "abstain", None
    if text == UNANSWERABLE_SENTINEL:
        return "unanswerable", None
    if text.endswith("%"):
        try:
            return "percent", round(float(text[:-1].replace(",", ".").replace(" ", "")), 2)
        except Exception:
            pass
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return "date", text
    if re.fullmatch(r"\d{4}-\d{2}", text):
        return "date", text

    numeric = parse_vn_number(text)
    if not math.isnan(numeric) and re.fullmatch(r"[+-]?[\d.,\s]+%?", text):
        rounded = _round_number(numeric)
        if rounded is not None:
            return "number", rounded

    return "string", text


PERCENT_HINTS = ("phần trăm", "tỉ lệ", "tỷ lệ", "tỉ trọng", "tỷ trọng", "%")
YES_NO_HINTS = ("có phải", "đúng không", "phải không", "hay không", "không?", "chưa?")


def apply_type_hints(kind: str, answer: Any, question: str) -> Tuple[str, Any]:
    """Sua type theo cau hoi de tranh mat diem do sai format (number vs percent vs boolean)."""
    q = question.lower()
    if kind == "number" and any(h in q for h in PERCENT_HINTS):
        return "percent", answer
    if kind == "string" and isinstance(answer, str):
        low = answer.lower()
        if low in ("true", "có", "đúng", "yes") and any(h in q for h in YES_NO_HINTS):
            return "boolean", True
        if low in ("false", "không", "sai", "no") and any(h in q for h in YES_NO_HINTS):
            return "boolean", False
    return kind, answer


# Cau hoi hoi NHIEU gia tri -> giu list
MULTI_ANSWER_HINTS = (
    "liệt kê", "liet ke", "danh sách", "danh sach", "top ", "top5", "top3",
    "những ", "các ", "lần lượt", "so sánh", "từng ", "mỗi ",
)
# Cau hoi hoi MOT gia tri -> khong duoc tra list
SINGLE_ANSWER_HINTS = (
    "nào", " ai", "ai ", "bao nhiêu", "bao nhieu", "mấy", "giờ nào", "ngày nào",
    "tháng nào", "tuần nào", "quý nào", "năm nào", "cao nhất", "thấp nhất",
    "nhiều nhất", "ít nhất", "lớn nhất", "nhỏ nhất", "tốt nhất", "giỏi nhất",
)


def _wants_single_value(question: str) -> bool:
    q = f" {question.lower()} "
    if any(k in q for k in MULTI_ANSWER_HINTS):
        return False
    return any(k in q for k in SINGLE_ANSWER_HINTS)


def collapse_to_single(kind: str, answer: Any, question: str) -> Tuple[str, Any]:
    """Cau hoi chi hoi 1 gia tri ma model tra list -> lay phan tu dau.

    Model gan nhu luon sap xep giam dan khi tu tao danh sach "top N", nen phan tu
    dau la dap an dung. Tra list cho cau hoi 1 gia tri la sai format (-1 diem).
    """
    if kind != "list" or not isinstance(answer, list) or not answer:
        return kind, answer
    if not _wants_single_value(question):
        return kind, answer

    first = answer[0]
    if isinstance(first, bool):
        return "boolean", first
    if isinstance(first, (int, float)):
        return "number", first
    if isinstance(first, str):
        return "string", first
    return kind, answer


# --------------------------------------------------------------------------- #
# 7. Nap model (unsloth -> transformers + peft)
# --------------------------------------------------------------------------- #
class Generator:
    def __init__(self, model, tokenizer, torch_module, backend: str,
                 max_prompt_tokens: int, max_new_tokens: int, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch_module
        self.backend = backend
        self.max_prompt_tokens = max_prompt_tokens
        self.max_new_tokens = max_new_tokens
        self.device = device
        self._prompt_len = 0
        try:
            # Neu prompt bi cat bot thi giu phan cuoi (cau hoi + chi dan), khong cat mat cau hoi
            tokenizer.truncation_side = "left"
        except Exception:
            pass
        self.stopping_criteria = self._build_stopping_criteria()

    def _build_stopping_criteria(self):
        if self.backend == "unsloth":
            return None  # generate cua unsloth khong nhan stopping_criteria
        try:
            from transformers import StoppingCriteria, StoppingCriteriaList
        except Exception:
            return None

        outer = self
        tokenizer = self.tokenizer

        class _FenceStopping(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                generated = input_ids[0, outer._prompt_len:]
                if len(generated) > 24:
                    tail = tokenizer.decode(generated[-8:], skip_special_tokens=False)
                    if "```" in tail or "<|im_end|>" in tail:
                        return True
                return False

        return StoppingCriteriaList([_FenceStopping()])

    def build_prompt(self, messages: Sequence[Dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )

    def generate(self, messages: Sequence[Dict[str, str]]) -> str:
        prompt = self.build_prompt(messages)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_prompt_tokens,
        )
        self._prompt_len = inputs["input_ids"].shape[1]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        kwargs: Dict[str, Any] = dict(
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if self.stopping_criteria is not None:
            kwargs["stopping_criteria"] = self.stopping_criteria

        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        return self.tokenizer.decode(output[0][self._prompt_len:], skip_special_tokens=True)


def _hf_cache_dir() -> str:
    try:
        from huggingface_hub import constants

        return str(constants.HF_HUB_CACHE)
    except Exception:
        return str(Path.home() / ".cache" / "huggingface" / "hub")


def resolve_base_model(args, use_4bit: bool, adapter_dir: Path) -> str:
    """Ten repo base model can tai (giong logic trong build_generator)."""
    fp_base = "Qwen/Qwen2.5-Coder-7B-Instruct"
    base_from_config = _read_adapter_base(adapter_dir) or fp_base
    if args.base_model:
        return args.base_model
    if use_4bit:
        return base_from_config
    return fp_base if base_from_config.endswith("bnb-4bit") else base_from_config


def prefetch_assets(args) -> int:
    """Tai truoc base model + tokenizer voi thanh tien trinh ro rang."""
    try:
        import torch
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(f"❌ Thiếu thư viện: {exc}\n   pip install -r requirements-local.txt")

    adapter_dir = Path(args.adapter).expanduser().resolve()
    cuda_ok = torch.cuda.is_available() and args.device != "cpu"
    use_4bit = args.load_in_4bit and cuda_ok and _bnb_available()
    base_name = resolve_base_model(args, use_4bit, adapter_dir)

    log(f"📦 Base model : {base_name}")
    log(f"📦 4-bit      : {use_4bit}")
    log(f"📦 Cache      : {_hf_cache_dir()}")
    log("⏳ Đang tải (có thể resume nếu ngắt giữa chừng)...\n")

    started = time.time()
    try:
        path = snapshot_download(repo_id=base_name, resume_download=True)
    except Exception as exc:
        log(f"❌ Tải thất bại: {type(exc).__name__}: {exc}")
        log("   Kiểm tra mạng, hoặc đặt HF_TOKEN nếu repo cần quyền.")
        return 1

    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    log(f"\n✅ Xong sau {(time.time() - started) / 60:.1f} phút: {path}")
    log(f"   Dung lượng: {total / 1024 ** 3:.2f} GB")
    log("   Giờ có thể chạy: python datamate_inference_local.py --output submission_local.jsonl")
    return 0


def _read_adapter_base(adapter_dir: Path) -> Optional[str]:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        return None
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        return cfg.get("base_model_name_or_path")
    except Exception:
        return None


def _bnb_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401
        return True
    except Exception:
        return False


def _load_tokenizer(adapter_dir: Path, base_name: str, fallback_name: str):
    from transformers import AutoTokenizer

    last_error: Optional[Exception] = None
    for source in (str(adapter_dir), base_name, fallback_name):
        try:
            tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
            break
        except Exception as exc:
            last_error = exc
            tokenizer = None
    if tokenizer is None:
        raise RuntimeError(f"Không nạp được tokenizer: {last_error}")

    if getattr(tokenizer, "chat_template", None) is None:
        template_path = adapter_dir / "chat_template.jinja"
        if template_path.exists():
            tokenizer.chat_template = template_path.read_text(encoding="utf-8")
            log("   ℹ️  Đã nạp chat_template từ chat_template.jinja")
        else:
            log("   ⚠️  Tokenizer không có chat_template, dùng template Qwen mặc định")
            tokenizer.chat_template = (
                "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
                "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
            )
    return tokenizer


def build_generator(args) -> Optional[Generator]:
    if args.device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    try:
        import torch
    except ImportError:
        raise SystemExit(
            "❌ Chưa cài PyTorch.\n"
            "   pip install -r requirements-local.txt\n"
            "   (hoặc chạy --dry-run để kiểm tra dữ liệu mà không cần model)"
        )

    cuda_ok = torch.cuda.is_available() and args.device != "cpu"
    adapter_dir = Path(args.adapter).expanduser().resolve()
    if not adapter_dir.exists():
        raise SystemExit(f"❌ Không tìm thấy LoRA adapter: {adapter_dir}")

    base_from_config = _read_adapter_base(adapter_dir) or "Qwen/Qwen2.5-Coder-7B-Instruct"
    fp_base = "Qwen/Qwen2.5-Coder-7B-Instruct"

    use_4bit = args.load_in_4bit and cuda_ok
    if use_4bit and not _bnb_available():
        log("⚠️  Chưa cài bitsandbytes -> chuyển sang fp16/bf16 (tốn VRAM hơn).")
        use_4bit = False

    if args.base_model:
        base_name = args.base_model
    elif use_4bit:
        base_name = base_from_config  # đúng base gốc mà adapter được train
    else:
        base_name = fp_base if base_from_config.endswith("bnb-4bit") else base_from_config

    log(f"🔧 Thiết bị: {'CUDA - ' + torch.cuda.get_device_name(0) if cuda_ok else 'CPU'}")
    log(f"🔧 Base model: {base_name}")
    log(f"🔧 LoRA adapter: {adapter_dir}")
    log(f"🔧 4-bit: {use_4bit}")

    if args.backend in ("auto", "unsloth") and cuda_ok:
        try:
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(adapter_dir),
                max_seq_length=args.max_seq_length,
                load_in_4bit=use_4bit,
                dtype=None,
            )
            FastLanguageModel.for_inference(model)
            log("✅ Nạp model bằng Unsloth (nhanh nhất).")
            return Generator(
                model, tokenizer, torch, "unsloth",
                args.max_prompt_tokens, args.max_new_tokens,
                device="cuda",
            )
        except Exception as exc:
            if args.backend == "unsloth":
                raise
            # Binh thuong tren Windows: Unsloth can Linux/WSL + triton.
            # transformers+peft van chay GPU binh thuong, chi cham hon ~1.5-2 lan.
            log(
                f"ℹ️  Không có Unsloth ({type(exc).__name__}) — bình thường trên Windows.\n"
                "    Dùng transformers+peft, vẫn chạy trên GPU."
            )

    return _build_transformers_generator(args, torch, adapter_dir, base_name, fp_base, cuda_ok, use_4bit)


def _build_transformers_generator(args, torch, adapter_dir, base_name, fp_base, cuda_ok, use_4bit):
    try:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
    except ImportError as exc:
        raise SystemExit(
            f"❌ Thiếu thư viện: {exc}\n   pip install -r requirements-local.txt"
        )

    tokenizer = _load_tokenizer(adapter_dir, base_name, fp_base)

    model_kwargs: Dict[str, Any] = dict(trust_remote_code=True, low_cpu_mem_usage=True)
    if cuda_ok:
        model_kwargs["device_map"] = "auto"
        if use_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["device_map"] = {"": "cpu"}
        model_kwargs["torch_dtype"] = torch.float32

    if args.dtype != "auto" and not use_4bit:
        model_kwargs["torch_dtype"] = getattr(torch, args.dtype)

    # Kiem tra VRAM truoc khi nap: game/browser co the dang chiem GPU
    vram_note = ""
    if cuda_ok:
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            free_gb, total_gb = free_b / 1024 ** 3, total_b / 1024 ** 3
            need_gb = 6.5 if use_4bit else 15.0
            vram_note = f"    VRAM: trống {free_gb:.1f}/{total_gb:.1f} GB (model cần ~{need_gb:.0f} GB)\n"
            if free_gb < need_gb:
                log(
                    f"⚠️  VRAM trống chỉ {free_gb:.1f} GB, model cần ~{need_gb:.0f} GB.\n"
                    "    Game/trình duyệt đang chiếm GPU (kiểm tra: nvidia-smi).\n"
                    "    Nên đóng bớt trước khi chạy, nếu không sẽ lỗi hoặc chậm gấp nhiều lần."
                )
        except Exception:
            pass

    log(
        "⏳ Đang nạp base model. Lần đầu phải TẢI ~4-5GB từ HuggingFace "
        "(có thể mất 5-20 phút tùy mạng), các lần sau lấy từ cache.\n"
        "    Muốn thấy thanh tiến trình tải: chạy `--prefetch` trước.\n"
        f"    Cache: {_hf_cache_dir()}\n{vram_note}".rstrip()
    )
    try:
        base_model = AutoModelForCausalLM.from_pretrained(base_name, **model_kwargs)
    except (ValueError, RuntimeError) as exc:
        # Thuong gap: game giu VRAM -> transformers phai offload CPU/disk -> bnb bao loi
        message = str(exc)
        if "dispatched on the CPU" in message or "out of memory" in message.lower():
            free_txt = ""
            try:
                free_b, total_b = torch.cuda.mem_get_info()
                free_txt = f" (trống {free_b / 1024 ** 3:.1f}/{total_b / 1024 ** 3:.1f} GB)"
            except Exception:
                pass
            raise SystemExit(
                f"❌ Không đủ VRAM để nạp model{free_txt}.\n"
                "   Cách xử lý:\n"
                "   1. Đóng game / ứng dụng đang dùng GPU rồi chạy lại (xem: nvidia-smi)\n"
                "   2. Hoặc giảm bộ nhớ: --max-prompt-tokens 768 --max-new-tokens 160\n"
                "   3. Hoặc chạy CPU (rất chậm): --device cpu"
            )
        if base_name == fp_base:
            raise
        log(f"⚠️  Không nạp được {base_name} ({type(exc).__name__}) -> thử {fp_base}")
        model_kwargs.pop("quantization_config", None)
        model_kwargs["torch_dtype"] = torch.bfloat16 if cuda_ok else torch.float32
        base_model = AutoModelForCausalLM.from_pretrained(fp_base, **model_kwargs)

    log("⏳ Đang gắn LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=False)
    if args.merge_lora:
        log("⏳ Merge LoRA vào base model...")
        model = model.merge_and_unload()
    model.eval()

    device = getattr(model, "device", None) or ("cuda" if cuda_ok else "cpu")
    device = str(device)
    log("✅ Nạp model bằng transformers + peft.")
    return Generator(
        model, tokenizer, torch, "transformers",
        args.max_prompt_tokens, args.max_new_tokens, device=device,
    )


# --------------------------------------------------------------------------- #
# 8. Prompt
# --------------------------------------------------------------------------- #
def build_messages(biz_id: str, query: str, as_of_date: str, schema: str) -> List[Dict[str, str]]:
    hint = resolve_time(query, as_of_date)
    period = resolve_period(query, as_of_date)
    period_line = ""
    if period is not None:
        start, end = period
        period_line = (
            f"Kỳ cần tính: {start.isoformat()} → {end.isoformat()} "
            "(lọc theo cột ngày đặt/ngày phát sinh)\n"
        )
    user_prompt = (
        f"Doanh nghiệp: {biz_id}\n"
        f"Ngày neo (as_of_date): {as_of_date} ({hint})\n"
        f"{period_line}"
        f"Schema các bảng dữ liệu sẵn có (trong từ điển `tables` hoặc biến `df`):\n{schema}\n\n"
        f'Câu hỏi: "{query}"\n\n'
        "Nhắc lại: loại đơn huỷ/trả hàng (giữ đơn đang giao), lọc đúng kỳ ở trên, "
        "chỉ dùng cột có trong schema.\n"
        "Viết mã Python pandas lưu kết quả cuối cùng vào biến `result` (đặt trong ```python ... ```):"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_messages_with_repair(
    biz_id: str, query: str, as_of_date: str, schema: str,
    previous_code: str, error: str,
) -> List[Dict[str, str]]:
    messages = build_messages(biz_id, query, as_of_date, schema)
    messages.append({"role": "assistant", "content": f"```python\n{previous_code}\n```"})
    messages.append({
        "role": "user",
        "content": (
            f"Đoạn mã trên lỗi: {error}\n"
            "Hãy sửa lại và trả về DUY NHẤT một khối ```python ... ``` gán kết quả vào biến `result`."
        ),
    })
    return messages


# --------------------------------------------------------------------------- #
# 9. Doc cau hoi
# --------------------------------------------------------------------------- #
def load_questions(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_existing_records(output_path: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    if not output_path.exists():
        return records
    with output_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if isinstance(record, dict) and "id" in record:
                    records[str(record["id"])] = record
            except json.JSONDecodeError:
                continue
    return records


def write_submission(questions: Sequence[Dict[str, Any]], records: Dict[str, Dict[str, Any]], path: Path) -> int:
    """Ghi lai file theo dung thu tu cau hoi, bo record trung."""
    seen = set()
    written = 0
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        for question in questions:
            qid = str(question["id"])
            if qid in seen:
                continue
            seen.add(qid)
            record = records.get(qid)
            if record is None:
                continue
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    tmp_path.replace(path)
    return written


# --------------------------------------------------------------------------- #
# 10. Verify
# --------------------------------------------------------------------------- #
_VALID_TYPES = {"number", "percent", "date", "boolean", "string", "list", "table", "unanswerable", "abstain"}


def verify_submission(questions: Sequence[Dict[str, Any]], path: Path) -> int:
    records = load_existing_records(path)
    problems: List[str] = []
    counts: Dict[str, int] = {}

    for question in questions:
        qid = str(question["id"])
        record = records.get(qid)
        if record is None:
            problems.append(f"{qid}: thiếu dòng")
            continue
        kind = str(record.get("type") or "abstain")
        counts[kind] = counts.get(kind, 0) + 1
        if kind not in _VALID_TYPES:
            problems.append(f"{qid}: type không hợp lệ `{kind}`")
            continue
        if kind in ("number", "percent"):
            value = record.get("answer")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                problems.append(f"{qid}: type={kind} nhưng answer={value!r} không phải số")
        elif kind == "boolean":
            if not isinstance(record.get("answer"), bool):
                problems.append(f"{qid}: type=boolean nhưng answer={record.get('answer')!r}")
        elif kind == "date":
            value = record.get("answer")
            if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", value or ""):
                problems.append(f"{qid}: type=date nhưng answer={value!r}")
        elif kind == "string":
            if not isinstance(record.get("answer"), str) or not record.get("answer", "").strip():
                problems.append(f"{qid}: type=string nhưng answer={record.get('answer')!r}")
        elif kind in ("abstain", "unanswerable"):
            if "answer" in record and record["answer"] is not None:
                problems.append(f"{qid}: type={kind} nhưng có answer")

    extra = set(records) - {str(q["id"]) for q in questions}
    for qid in sorted(extra):
        problems.append(f"{qid}: dòng thừa (không có trong questions.jsonl)")

    log(f"\n📊 {path}")
    log(f"   Câu hỏi       : {len(questions)}")
    log(f"   Dòng hợp lệ   : {len(questions) - len([p for p in problems if 'thiếu dòng' in p])}")
    log(f"   Theo type     : {dict(sorted(counts.items(), key=lambda kv: -kv[1]))}")
    answered = sum(v for k, v in counts.items() if k not in ("abstain", "unanswerable"))
    log(f"   Có đáp án     : {answered}  |  Abstain: {counts.get('abstain', 0)}")
    if problems:
        log(f"\n⚠️  {len(problems)} vấn đề (hiển thị tối đa 25):")
        for problem in problems[:25]:
            log(f"   - {problem}")
    else:
        log("✅ File submission hợp lệ về format và đủ id.")
    return 0 if not problems else 1


# --------------------------------------------------------------------------- #
# 11. Vong lap chinh
# --------------------------------------------------------------------------- #
def run(args) -> int:
    data_root = Path(args.data_root).expanduser().resolve()
    questions_path = Path(args.questions).expanduser().resolve() if args.questions else data_root / "questions.jsonl"
    business_root = data_root / "data"
    output_path = Path(args.output).expanduser().resolve()

    if not questions_path.exists():
        raise SystemExit(f"❌ Không tìm thấy questions.jsonl: {questions_path}")
    if not business_root.exists():
        raise SystemExit(f"❌ Không tìm thấy thư mục data: {business_root}")

    questions = load_questions(questions_path)
    all_questions = list(questions)  # giu danh sach day du de ghi file cuoi cung
    if args.business:
        wanted = {b.strip() for b in args.business.split(",") if b.strip()}
        questions = [q for q in questions if q.get("business_id") in wanted]
    if args.ids:
        wanted_ids = {i.strip() for i in args.ids.split(",") if i.strip()}
        questions = [q for q in questions if q.get("id") in wanted_ids]
    if args.offset:
        questions = questions[args.offset:]
    if args.limit:
        questions = questions[: args.limit]

    if not questions:
        raise SystemExit("❌ Không có câu hỏi nào sau khi lọc.")

    log(f"📄 Dữ liệu : {data_root}")
    log(f"📄 Câu hỏi : {len(questions)} (từ {questions_path.name})")

    calendar = load_calendar(data_root)
    table_cache = TableCache(business_root, max_items=args.table_cache)

    generator: Optional[Generator] = None
    if not args.dry_run:
        generator = build_generator(args)
    else:
        log("🧪 Chế độ --dry-run: bỏ qua nạp model.")

    existing: Dict[str, Dict[str, Any]] = {}
    if output_path.exists() and not args.overwrite:
        existing = load_existing_records(output_path)
        hit = sum(1 for q in questions if str(q["id"]) in existing)
        if hit:
            log(f"🔄 Resume: {hit}/{len(questions)} câu đã có sẵn trong {output_path.name}")
    elif args.overwrite and output_path.exists():
        log(f"🗑️  Ghi đè {output_path.name} (--overwrite)")
        try:
            output_path.unlink()
        except PermissionError:
            # File đang bị process khác giữ -> ghi rỗng thay vì crash
            try:
                output_path.write_text("", encoding="utf-8")
            except Exception as exc:
                raise SystemExit(f"❌ Không ghi được {output_path}: {exc}")

    records: Dict[str, Dict[str, Any]] = dict(existing)
    if args.retry_abstain:
        pending = [
            q for q in questions
            if str(q["id"]) not in existing
            or str(existing[str(q["id"])].get("type")) in ("abstain", "unanswerable")
        ]
    else:
        pending = [q for q in questions if str(q["id"]) not in existing]

    log(f"🚀 Bắt đầu suy luận: {len(pending)} câu cần chạy\n")

    start_time = time.time()
    processed = 0
    failed = 0
    timeouts = 0
    type_counts: Dict[str, int] = {}

    out_handle = _NullWriter() if args.dry_run else output_path.open("a", encoding="utf-8")
    with contextlib.closing(out_handle) as out_fh:
        for index, question in enumerate(pending, start=1):
            qid = str(question["id"])
            biz_id = str(question.get("business_id", ""))
            query = str(question.get("question", ""))
            as_of_date = str(question.get("as_of_date", "2026-09-07"))

            try:
                biz_tables = table_cache.get(biz_id)
                if not biz_tables:
                    log(f"⚠️  {qid}: không có dữ liệu cho {biz_id}")
            except Exception as exc:
                log(f"⚠️  {qid}: lỗi đọc dữ liệu {biz_id}: {exc}")
                biz_tables = {}

            sandbox_tables: Dict[str, pd.DataFrame] = dict(biz_tables)
            if calendar is not None:
                sandbox_tables["calendar"] = calendar

            schema, default_table = build_schema_within_budget(
                query=query,
                tables=biz_tables,
                calendar=calendar,
                max_chars=int(args.max_prompt_tokens * 2.2),
                top_k=args.schema_top_k,
                sample_rows=args.schema_sample_rows,
                as_of_date=as_of_date,
            )

            if args.dry_run:
                messages = build_messages(biz_id, query, as_of_date, schema)
                log(
                    f"   [{qid}] {biz_id} | df={default_table} | "
                    f"{len(biz_tables)} bảng | schema {len(schema)} ký tự"
                )
                if index == 1:
                    log("─" * 70)
                    log(f"VÍ DỤ PROMPT ({qid})\n{messages[1]['content']}")
                    log("─" * 70)
                continue

            assert generator is not None, "Model chưa được nạp (chỉ xảy ra khi --dry-run)"
            record = None
            last_code = ""
            last_error = "chưa chạy"
            for attempt in range(1, args.max_attempts + 1):
                if attempt == 1:
                    messages = build_messages(biz_id, query, as_of_date, schema)
                else:
                    messages = build_messages_with_repair(
                        biz_id, query, as_of_date, schema, last_code, last_error
                    )

                try:
                    raw_response = generator.generate(messages)
                except Exception as exc:
                    last_error = f"generate_failed: {type(exc).__name__}: {exc}"
                    log(f"⚠️  {qid}: lỗi generate ({type(exc).__name__}: {exc})")
                    continue

                last_code = extract_code(raw_response)

                # Chan model bia danh sach ten san pham/khach roi loc theo do
                hallucinated: List[str] = []
                if args.literal_guard:
                    hallucinated = find_hallucinated_literals(
                        last_code, sandbox_tables, default_table
                    )
                if hallucinated:
                    last_error = (
                        "Các giá trị sau KHÔNG tồn tại trong dữ liệu: "
                        + ", ".join(hallucinated[:5])
                        + ". Không được hardcode danh sách; hãy lọc từ dữ liệu bằng "
                        ".str.contains(...), isin(df['cột'].unique()) hoặc groupby."
                    )
                    if attempt < args.max_attempts:
                        log(f"   ⚠️  {qid}: code hardcode giá trị không có trong dữ liệu -> sửa")
                        continue
                    ok, value, error = False, None, last_error
                else:
                    ok, value, error = execute_sandbox(
                        raw_response, sandbox_tables, as_of_date,
                        timeout_s=args.sandbox_timeout, default_table=default_table,
                    )
                if error.startswith("Code chạy quá"):
                    timeouts += 1
                last_error = error

                if args.verbose and processed < args.verbose:
                    log(f"   [{qid}] mã sinh ra:\n{last_code}\n   Lỗi: {error or 'không'}")

                if ok:
                    kind, answer = infer_type(value)
                    if args.type_hints:
                        kind, answer = apply_type_hints(kind, answer, query)
                        kind, answer = collapse_to_single(kind, answer, query)
                    if kind in ("abstain", "unanswerable"):
                        record = {"id": qid, "type": kind}
                    else:
                        record = {"id": qid, "type": kind, "answer": answer}
                    break

            if record is None:
                failed += 1
                record = {"id": qid, "type": "abstain"}
                if args.show_errors:
                    log(f"❌ {qid}: {last_error}")

            records[qid] = record
            type_counts[record["type"]] = type_counts.get(record["type"], 0) + 1
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_fh.flush()

            processed += 1
            if processed % args.log_every == 0 or processed == len(pending):
                elapsed = time.time() - start_time
                speed = elapsed / processed
                remaining = (len(pending) - processed) * speed / 60
                answered = sum(v for k, v in type_counts.items() if k not in ("abstain", "unanswerable"))
                log(
                    f"🔥 {processed}/{len(pending)} | {speed:.2f}s/câu | còn ~{remaining:.1f} phút "
                    f"| có đáp án {answered}/{processed} | abstain {processed - answered} "
                    f"| lỗi {failed} | timeout {timeouts}"
                )

    if args.dry_run:
        log("\n✅ Dry-run xong: dữ liệu và prompt đều ổn.")
        return 0

    written = write_submission(all_questions, records, output_path)
    elapsed = time.time() - start_time
    log(f"\n🎉 Hoàn thành: {written}/{len(all_questions)} câu -> {output_path}")
    log(f"   Thời gian: {elapsed / 60:.1f} phút | lỗi: {failed} | timeout: {timeouts}")

    stats_path = output_path.with_suffix(".stats.json")
    stats = {
        "output": str(output_path),
        "questions": len(questions),
        "written": written,
        "elapsed_minutes": round(elapsed / 60, 2),
        "failed_codegen": failed,
        "timeouts": timeouts,
        "type_counts": type_counts,
        "backend": args.backend,
        "adapter": str(args.adapter),
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
    }
    try:
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"   Thống kê: {stats_path.name}")
    except Exception:
        pass

    verify_submission(all_questions, output_path)
    return 0


# --------------------------------------------------------------------------- #
# 12. CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DataMate 2026 - chạy suy luận local (không cần Colab).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    paths = parser.add_argument_group("Đường dẫn")
    paths.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Thư mục chứa questions.jsonl + data/")
    paths.add_argument("--questions", default=None, help="Đường dẫn questions.jsonl (mặc định <data-root>/questions.jsonl)")
    paths.add_argument("--adapter", default=str(DEFAULT_ADAPTER), help="Thư mục LoRA adapter")
    paths.add_argument("--base-model", default=None, help="Override base model HF id / đường dẫn local")
    paths.add_argument("--output", default=str(DEFAULT_OUTPUT), help="File submission đầu ra")
    paths.add_argument("--log-file", default=None, help="Ghi log ra file")

    subset = parser.add_argument_group("Chạy thử / lọc")
    subset.add_argument("--limit", type=int, default=0, help="Chỉ chạy N câu đầu")
    subset.add_argument("--offset", type=int, default=0, help="Bỏ qua N câu đầu")
    subset.add_argument("--business", default=None, help="Lọc theo business_id, cách nhau dấu phẩy")
    subset.add_argument("--ids", default=None, help="Lọc theo id câu hỏi, cách nhau dấu phẩy")
    subset.add_argument("--overwrite", action="store_true", help="Xoá file output cũ, không resume")
    subset.add_argument("--retry-abstain", action="store_true", help="Chạy lại cả các câu đang là abstain")
    subset.add_argument("--dry-run", action="store_true", help="Chỉ kiểm tra dữ liệu + prompt, không nạp model")
    subset.add_argument("--verify-only", action="store_true", help="Chỉ kiểm tra format file submission rồi thoát")
    subset.add_argument("--prefetch", action="store_true", help="Tải trước base model về cache (có thanh tiến trình)")

    model = parser.add_argument_group("Model")
    model.add_argument("--backend", choices=["auto", "unsloth", "transformers", "cpu"], default="auto")
    model.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    model.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    model.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true", default=True)
    model.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    model.add_argument("--merge-lora", action="store_true", help="Merge LoRA vào base (chậm hơn khi nạp, nhanh hơn khi generate)")
    model.add_argument("--max-seq-length", type=int, default=2048)
    model.add_argument("--max-prompt-tokens", type=int, default=1536)
    model.add_argument("--max-new-tokens", type=int, default=320)

    perf = parser.add_argument_group("Hiệu năng & sandbox")
    perf.add_argument("--sandbox-timeout", type=float, default=8.0, help="Giới hạn thời gian 1 đoạn code (giây)")
    perf.add_argument("--max-attempts", type=int, default=2, help="Số lần thử lại khi code lỗi")
    perf.add_argument("--schema-top-k", type=int, default=4, help="Số bảng tối đa đưa vào prompt (0 = tất cả)")
    perf.add_argument("--schema-sample-rows", type=int, default=2, help="Số dòng mẫu mỗi bảng trong prompt")
    perf.add_argument("--table-cache", type=int, default=0, help="Số doanh nghiệp giữ trong RAM (0 = tất cả)")
    perf.add_argument("--log-every", type=int, default=10)
    perf.add_argument("--no-literal-guard", dest="literal_guard", action="store_false",
                      default=True, help="Tắt kiểm tra hardcode danh sách tên")

    misc = parser.add_argument_group("Log")
    misc.add_argument("--verbose", type=int, default=3, help="In mã sinh ra cho N câu đầu")
    misc.add_argument("--show-errors", action="store_true", help="In lỗi của các câu bị lỗi")
    misc.add_argument("--no-type-hints", dest="type_hints", action="store_false", default=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    global _LOG_PATH
    _setup_console()
    args = build_parser().parse_args(argv)

    if args.log_file:
        _LOG_PATH = Path(args.log_file).expanduser().resolve()
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    if args.prefetch:
        return prefetch_assets(args)

    if args.verify_only:
        data_root = Path(args.data_root).expanduser().resolve()
        questions_path = Path(args.questions).expanduser().resolve() if args.questions else data_root / "questions.jsonl"
        questions = load_questions(questions_path)
        return verify_submission(questions, Path(args.output).expanduser().resolve())

    try:
        return run(args)
    except KeyboardInterrupt:
        log("\n⏸️  Đã dừng bằng Ctrl+C. File output vẫn giữ các câu đã xong, chạy lại sẽ resume.")
        return 130
    except SystemExit:
        raise
    except Exception:
        log("\n💥 Lỗi không mong đợi:")
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
