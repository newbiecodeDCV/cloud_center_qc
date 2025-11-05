import base64
import hashlib
import json
import urllib.parse

import magic


def hash_str(string: str, n_hash: int = 9) -> int:
    """
    Hàm băm (hash) một chuỗi thành số nguyên có độ dài n_hash chữ số.

    Tham số:
    - string: Chuỗi đầu vào cần băm.
    - n_hash: Số lượng chữ số mong muốn của số nguyên kết quả (mặc định là 9).

    Trả về:
    - Số nguyên đại diện cho mã băm của chuỗi đầu vào.
    """
    return int(hashlib.sha256(string.encode("utf-8")).hexdigest(), 16) % 10**n_hash


def create_task_id(audio_bytes: bytes = None, url: str = None):
    """
    Tạo mã task_id duy nhất dựa trên dữ liệu âm thanh (dạng bytes, sẽ được encode base64)
    hoặc từ URL. Mã task_id dạng số nguyên 6 chữ số.

    Tham số:
    - audio_bytes: (tùy chọn) Dữ liệu âm thanh dạng bytes.
    - url: (tùy chọn) Đường dẫn URL đến file âm thanh.

    Trả về:
    - Số nguyên task_id (10 chữ số).
    """
    if audio_bytes:
        # NOTE(by ducnt2): using base64 format to avoid errors when hassing
        audio_bytes = base64.b64encode(audio_bytes).decode("ascii")
        audio_id = str(audio_bytes)
        audio_id = hash_str(audio_id)
    elif url:
        audio_id = hash_str(url)
    else:
        raise ValueError("Cần cung cấp 'audio_bytes' hoặc 'url' để tạo task_id")

    task_id = f"{audio_id}"
    task_id = int(f"{hash_str(task_id, 10)}")
    return task_id


def seconds_to_min_sec(seconds):
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes}:{secs:02d}"


def is_url(string):
    """
    Check if string is a valid url
    """
    try:
        result = urllib.parse.urlparse(string)
        return bool(result.scheme) and bool(result.netloc)
    except Exception:
        return False


def is_audio_file(file_bytes):
    mime = magic.Magic(mime=True)
    mime_type = mime.from_buffer(file_bytes)
    return mime_type.startswith("audio")


def write_json(data: dict, json_file_path: str):
    with open(json_file_path, "w") as file:
        json.dump(data, file, indent=4)
