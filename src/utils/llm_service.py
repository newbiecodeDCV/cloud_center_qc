import json
import os

import openai
from dotenv import load_dotenv

# Tải các biến từ file .env vào môi trường
load_dotenv()

# Lấy thông tin cấu hình từ biến môi trường
API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("OPENAI_API_BASE")
MODEL_NAME = os.getenv("MODEL_NAME")

# Chỉ khởi tạo client khi có đủ cấu hình, tránh raise khi import
client = None
if API_KEY and API_BASE_URL:
    client = openai.AsyncOpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
    )

# Khởi tạo client OpenAI với cấu hình tùy chỉnh
client = openai.AsyncOpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL,
)


def build_prompt(call_data: dict) -> str:
    """
    Xây dựng chuỗi prompt hoàn chỉnh dựa trên file tiêu chí chấm điểm CSV và dữ liệu cuộc gọi.
    """
    call_data_str = json.dumps(call_data, indent=2, ensure_ascii=False)

    prompt_template = f"""
Bạn là một chuyên gia đánh giá chất lượng (QA) cuộc gọi cực kỳ chi tiết. Nhiệm vụ của bạn là chấm điểm cuộc gọi dựa trên các tiêu chí trong nhóm "Kỹ năng giao tiếp" được trích xuất từ file quy định và các dữ liệu âm học đã được phân tích.

**QUY TẮC BẮT BUỘC:**
1.  **TUÂN THỦ BỘ TIÊU CHÍ**: Chỉ chấm điểm dựa trên các tiêu chí con được liệt kê dưới đây.
2.  **BỎ QUA NGHIỆP VỤ**: Hoàn toàn không đánh giá nội dung về sản phẩm, bán hàng, hay xử lý khiếu nại. Tập trung 100% vào kỹ năng giao tiếp thể hiện qua giọng nói và cấu trúc hội thoại.
3.  **DỰA VÀO BẰNG CHỨNG ÂM HỌC**: Mọi nhận xét phải bắt nguồn từ các chỉ số trong "DỮ LIỆU CUỘC GỌI".

**BỘ TIÊU CHÍ CHẤM ĐIỂM - NHÓM 1: KỸ NĂNG GIAO TIẾP:**

* **Tiêu chí 1.1: Chào/ Xưng danh**
    * **Mô tả**: Nhân viên có chào hỏi và xưng danh đầy đủ, thân thiện ở đầu cuộc gọi không?
    * **Dữ liệu tham khảo**: `text` của các segment đầu tiên.

* **Tiêu chí 1.2: Kỹ năng nói (Tốc độ - Âm lượng)**
    * **Mô tả**: Đánh giá tốc độ nói có phù hợp, rõ ràng không. Âm lượng có đủ nghe, ổn định không.
    * **Dữ liệu tham khảo**: `speed_wpm` (Tốc độ tốt: 120-160 WPM), `volume_db` (Âm lượng tốt: ổn định và không quá thấp, ví dụ trên -50dB).

* **Tiêu chí 1.3: Kỹ năng nghe, Trấn an**
    * **Mô tả**: Nhân viên có lắng nghe, không ngắt lời khách hàng? Thời gian nói có cân bằng không? Có các khoảng lặng để khách hàng trình bày không?
    * **Dữ liệu tham khảo**: `metadata.turns` (số lượt nói càng cao càng tốt), `metadata.ratio_sales` (tỷ lệ nói của nhân viên, mức tốt là 50-70%), `silence_ratio` trong các segment của khách hàng.

* **Tiêu chí 1.4: Thái độ giao tiếp**
    * **Mô tả**: Thái độ có nhiệt tình, thân thiện, chuyên nghiệp không? Giọng nói có truyền cảm hay đơn điệu?
    * **Dữ liệu tham khảo**: `pitch_hz` (sự biến thiên cao độ cho thấy sự nhiệt tình, truyền cảm), `volume_db` (sự ổn định cho thấy thái độ chuyên nghiệp).

**DỮ LIỆU CUỘC GỌI:**
```json
{call_data_str}
YÊU CẦU ĐẦU RA (OUTPUT FORMAT): Hãy trả về một đối tượng JSON duy nhất tuân thủ nghiêm ngặt cấu trúc sau.
{{
  "nhan_xet_tong_quan": "Cuộc gọi đạt chuẩn về kỹ năng giao tiếp, không phát hiện lỗi nghiêm trọng. Nhân viên cần chú ý điều chỉnh tốc độ nói ở một vài thời điểm để hoàn thiện hơn.",
  "danh_sach_loi_phat_hien": [
    "Mức 1 - Kỹ năng nói: Tốc độ nói hơi nhanh ở một vài phân đoạn."
  ],
  "chi_tiet_danh_gia": [
    {{
      "tieu_chi": "Chào/ Xưng danh",
      "trang_thai": "Đạt",
      "loi_phat_hien": "Không có",
      "bang_chung": "Nhân viên đã chào và xưng danh ở segment đầu tiên."
    }},
    {{
      "tieu_chi": "Kỹ năng nói (Tốc độ - Âm lượng)",
      "trang_thai": "Không Đạt",
      "loi_phat_hien": "Mức 1: Tốc độ nói có lúc chưa phù hợp.",
      "bang_chung": "Segment bắt đầu tại 88.2s có speed_wpm là 172, cao hơn mức khuyến nghị."
    }},
    {{
      "tieu_chi": "Kỹ năng nghe",
      "trang_thai": "Đạt",
      "loi_phat_hien": "Không có",
      "bang_chung": "Số lượt nói (turns) là 20 và tỷ lệ nói của nhân viên (ratio_sales) là 0.65, cho thấy sự lắng nghe tốt."
    }},
    {{
      "tieu_chi": "Thái độ giao tiếp",
      "trang_thai": "Đạt",
      "loi_phat_hien": "Không có",
      "bang_chung": "Sự biến thiên của pitch_hz trong các phát ngôn của nhân viên cho thấy giọng nói có năng lượng và nhiệt tình."
    }}
  ]
}}
"""
    return prompt_template


async def get_qa_evaluation(call_data: dict) -> dict:
    """Gửi prompt đến LLM và nhận kết quả đánh giá QA, tối ưu hóa cho LiteLLM."""
    prompt = build_prompt(call_data)

    try:
        if client is None:
            return {
                "error": "Thiếu cấu hình OPENAI_API_KEY/OPENAI_API_BASE trong môi trường"
            }
        if not MODEL_NAME:
            return {"error": "Thiếu MODEL_NAME trong môi trường"}
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert QA system. Your sole purpose is to return a valid, complete JSON object based on the user's request. Do not include any other text, explanations, or markdown formatting outside of the JSON structure.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        result_content = response.choices[0].message.content

        # Xử lý trường hợp model vẫn trả về khối mã markdown
        if result_content.strip().startswith("```json"):
            result_content = result_content.strip()[7:-3]

        return json.loads(result_content)

    except json.JSONDecodeError as json_err:
        print(f"Lỗi khi parse JSON: {json_err}")
        print(f"Nội dung nhận được từ model: {result_content}")
        return {
            "error": "Lỗi khi phân tích JSON từ model.",
            "raw_response": result_content,
        }
    except Exception as e:
        print(f"Lỗi khi gọi API của LLM: {e}")
        return {"error": str(e)}
