import openai
from dotenv import load_dotenv
import os
import json
from src.qa_communicate.prompt.prompts import build_qa_prompt

# Tải các biến từ file .env vào môi trường
load_dotenv()

# Lấy thông tin cấu hình từ biến môi trường
API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")

# Khởi tạo client OpenAI với cấu hình tùy chỉnh
if API_KEY and API_BASE_URL:
    client = openai.AsyncOpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL,
    )
else:
    client = None
    print("⚠️ CẢNH BÁO: Thiếu OPENAI_API_KEY hoặc OPENAI_API_BASE trong file .env")


async def get_qa_evaluation(call_data: dict) -> dict:
    """
    Gửi prompt đến LLM và nhận kết quả đánh giá QA.
    """
    print(f"DEBUG - API_KEY exists: {bool(API_KEY)}")
    print(f"DEBUG - API_BASE_URL: {API_BASE_URL}")
    print(f"DEBUG - MODEL_NAME: {MODEL_NAME}")
    prompt = build_qa_prompt(call_data)

    try:
        if client is None:
            return {"error": "Thiếu cấu hình OPENAI_API_KEY/OPENAI_API_BASE"}
        if not MODEL_NAME:
            return {"error": "Thiếu MODEL_NAME"}

        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "Bạn là chuyên gia phân tích chất lượng cuộc gọi. Đánh giá chính xác dựa trên dữ liệu âm học và transcript. Trả về JSON theo format đã cho."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        result_content = response.choices[0].message.content.strip()

        # Xử lý markdown block
        if result_content.startswith("```json"):
            result_content = result_content[7:]
        if result_content.endswith("```"):
            result_content = result_content[:-3]
        
        result_content = result_content.strip()

        return json.loads(result_content)

    except json.JSONDecodeError as json_err:
        print(f"Lỗi parse JSON: {json_err}")
        print(f"Nội dung từ model: {result_content}")
        return {
            "error": "Lỗi phân tích JSON từ model",
            "raw_response": result_content
        }
    except Exception as e:
        print(f"Lỗi khi gọi LLM API: {e}")
        return {"error": str(e)}