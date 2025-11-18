import argparse
import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Optional
import os

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.utils import hash_str, create_task_id
from src.audio_processing.dialogue import call_dialogue_api
from src.audio_processing.analysis import extract_features


PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def save_json_result(data: Any, prefix: str, output_dir: str = "results"):
    """Lưu kết quả ra file JSON với timestamp"""
    output_path = PROJECT_ROOT_DIR / output_dir / "dialogue"
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.json"
    filepath = output_path / filename

    result_with_metadata = {
        "timestamp": datetime.now().isoformat(),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M:%S"),
        "result": data,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result_with_metadata, f, ensure_ascii=False, indent=2)

    print(f"✓ Đã lưu kết quả vào: {filepath}")
    return filepath


def read_file_bytes(file_path: str) -> bytes:
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")
    return path.read_bytes()


async def maybe_call_dialogue(
    audio_bytes: bytes, task_id: int, *, timeout: float, verbose: bool, output_dir: str
):
    try:
        print("\n" + "=" * 60)
        print("🔄 Đang gọi call_dialogue_api...")
        print("=" * 60)

        result = await call_dialogue_api(
            audio_bytes=audio_bytes,
            task_id=task_id,
            max_poll_seconds=timeout,
            poll_interval_seconds=0.5,
            verbose=verbose,
        )

        print("\n" + "=" * 60)
        print("✅ API trả về thành công!")
        print("=" * 60)
        print("\nKết quả call_dialogue_api:")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        # Lưu kết quả API ra file JSON
        save_json_result(result, "dialogue_api_result", output_dir)

    except Exception as exc:
        print("\n" + "=" * 60)
        print("❌ Lỗi khi gọi call_dialogue_api!")
        print("=" * 60)
        print(f"Chi tiết lỗi: {exc}")

        # Lưu lỗi ra file JSON
        error_data = {"error": str(exc), "error_type": type(exc).__name__}
        save_json_result(error_data, "dialogue_api_error", output_dir)


async def analyze_audio(audio_bytes: bytes, output_dir: str):
    try:
        print("\n" + "=" * 60)
        print("🔊 Đang phân tích audio với extract_features...")
        print("=" * 60)

        result = await extract_features(audio_bytes)

        print("\n" + "=" * 60)
        print("✅ Phân tích audio hoàn tất!")
        print("=" * 60)
        print("\nKết quả extract_features:")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        # Lưu kết quả phân tích ra file JSON
        save_json_result(result, "audio_features_result", output_dir)

    except Exception as exc:
        print("\n" + "=" * 60)
        print("❌ Lỗi khi phân tích audio!")
        print("=" * 60)
        print(f"Chi tiết lỗi: {exc}")

        # Lưu lỗi ra file JSON
        error_data = {"error": str(exc), "error_type": type(exc).__name__}
        save_json_result(error_data, "audio_features_error", output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Kiểm thử utils, dialogue_utils và audio_analysis"
    )
    parser.add_argument(
        "--audio-file",
        type=str,
        default=None,
        help="Đường dẫn tới file WAV để test API",
    )
    parser.add_argument(
        "--audio-url",
        type=str,
        default=None,
        help="URL file audio để tạo task_id (không tải)",
    )
    parser.add_argument(
        "--analyze", action="store_true", help="Gọi extract_features để phân tích audio"
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="Timeout (giây) chờ kết quả API"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="In log polling chi tiết"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Thư mục lưu kết quả JSON (mặc định: results/)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(f"🧪 BẮT ĐẦU TEST - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Test hash_str
    print("\n📝 Test hash_str:")
    sample_text = "xin chao cloud callcenter qa"
    h9 = hash_str(sample_text, 9)
    h6 = hash_str(sample_text, 6)
    print(f"   Input: '{sample_text}'")
    print(f"   Hash (9 digits): {h9}")
    print(f"   Hash (6 digits): {h6}")

    # Chuẩn bị dữ liệu cho create_task_id
    audio_bytes = None
    if args.audio_file:
        try:
            print(f"\n📂 Đọc file audio: {args.audio_file}")
            audio_bytes = read_file_bytes(args.audio_file)
            file_size_mb = len(audio_bytes) / (1024 * 1024)
            print(f"   ✓ Đã đọc {len(audio_bytes)} bytes ({file_size_mb:.2f} MB)")
        except Exception as e:
            print(f"   ❌ Không đọc được file audio: {e}")

    # Test create_task_id
    task_id = None
    try:
        print("\n🔢 Tạo task_id:")
        task_id = create_task_id(audio_bytes=audio_bytes, url=args.audio_url)
        print(f"   ✓ task_id = {task_id}")
    except Exception as exc:
        print(f"   ❌ Lỗi khi tạo task_id: {exc}")

    # Gọi API nếu có audio_bytes và có task_id hợp lệ
    if audio_bytes is not None and isinstance(task_id, int):
        asyncio.run(
            maybe_call_dialogue(
                audio_bytes=audio_bytes,
                task_id=task_id,
                timeout=args.timeout,
                verbose=args.verbose,
                output_dir=args.output_dir,
            )
        )
    else:
        print("\n⚠️  Bỏ qua gọi API (cần '--audio-file' và 'task_id' hợp lệ).")

    # Phân tích audio nếu có cờ --analyze và có audio_bytes
    if args.analyze and audio_bytes is not None:
        asyncio.run(analyze_audio(audio_bytes, args.output_dir))
    elif args.analyze:
        print("\n⚠️  Bỏ qua phân tích audio (không có audio_bytes).")

    print("\n" + "=" * 60)
    print("🎉 HOÀN THÀNH!")
    print(f"📁 Các file kết quả được lưu trong thư mục: {args.output_dir}/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test bị dừng bởi người dùng")
        sys.exit(130)
