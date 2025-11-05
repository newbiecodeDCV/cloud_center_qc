import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Thêm thư mục src vào Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.audio_processing.analysis import extract_features
    from src.evaluation.evaluator import get_qa_evaluation
except ImportError as e:
    print(f"Lỗi Import: {e}")
    print("Hãy chắc chắn rằng bạn đang chạy script này từ thư mục gốc của dự án.")
    sys.exit(1)

# Tạo thư mục 'results' nếu chưa tồn tại
RESULTS_DIR = Path(__file__).parent.parent / "results" / "score"
os.makedirs(RESULTS_DIR, exist_ok=True)


def read_file_bytes(file_path: str) -> bytes:
    """Đọc nội dung file audio thành bytes."""
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file audio tại: {file_path}")
    print(f"Đã đọc file: {file_path}")
    return path.read_bytes()


async def main():
    """Hàm chính để chạy toàn bộ quy trình: Phân tích -> Chấm điểm -> Lưu kết quả."""
    if len(sys.argv) < 2:
        print("Lỗi: Vui lòng cung cấp đường dẫn đến file audio.")
        print("Cách dùng: python run_real_test.py <đường_dẫn_file_audio.wav>")
        sys.exit(1)

    audio_file_path = Path(sys.argv[1])

    try:
        # 1. Đọc file audio
        audio_bytes = read_file_bytes(str(audio_file_path))

        # 2. Phân tích đặc trưng âm học
        print("\nBước 1: Bắt đầu phân tích đặc trưng âm học...")
        analysis_result = await extract_features(audio_bytes)

        if analysis_result.get("status") != 1:
            print("Phân tích âm học thất bại. Dừng chương trình.")
            print(analysis_result.get("message"))
            return

        print("Phân tích âm học thành công!")

        # 3. Gửi dữ liệu tới LLM để chấm điểm
        print("\nBước 2: Gửi dữ liệu tới LLM để chấm điểm...")
        data_for_llm = {
            "metadata": analysis_result.get("metadata"),
            "segments": analysis_result.get("segments"),
        }

        evaluation_result = await get_qa_evaluation(data_for_llm)

        # 4. In và LƯU kết quả cuối cùng
        print("\n--- HOÀN TẤT! KẾT QUẢ CHẤM ĐIỂM TỪ LLM ---")
        if evaluation_result and "error" not in evaluation_result:
            result_str = json.dumps(evaluation_result, indent=4, ensure_ascii=False)
            print(result_str)

            # Tạo tên file output dựa trên tên file audio và thời gian
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{audio_file_path.stem}_evaluation_{timestamp}.json"
            output_path = RESULTS_DIR / output_filename

            # Lưu kết quả vào file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result_str)
            print(f"\n✅ Đã lưu kết quả thành công vào file: {output_path}")

        else:
            print("Không nhận được kết quả chấm điểm hoặc có lỗi xảy ra.")
            print(evaluation_result)

    except FileNotFoundError as e:
        print(f"Lỗi: {e}")
    except Exception as e:
        print(f"Đã xảy ra lỗi không mong muốn: {e}")


if __name__ == "__main__":
    asyncio.run(main())
