import openai
from dotenv import load_dotenv
import os
import json
from src.qa_communicate.prompt.prompts import build_qa_prompt
from src.qa_communicate.core.langfuse_config import (
    LANGFUSE_ENABLED,
    create_trace,
    log_generation,
    log_span
)




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


async def get_qa_evaluation(call_data: dict,task_id: int = None) -> dict:
    """
    Gửi prompt đến LLM và nhận kết quả đánh giá QA.
    """
    print(f"DEBUG - API_KEY exists: {bool(API_KEY)}")
    print(f"DEBUG - API_BASE_URL: {API_BASE_URL}")
    print(f"DEBUG - MODEL_NAME: {MODEL_NAME}")
    

    trace = None
    if LANGFUSE_ENABLED:
        trace = create_trace(
            name="evaluate_communication_skills",
            metadata={
                "task_id": task_id,
                "model": MODEL_NAME,
                "duration": call_data.get("metadata", {}).get("duration"),
                "turns": call_data.get("metadata", {}).get("turns")
            }
        )
    
    if trace:
        log_span(
            trace=trace,
            name="build_prompt",
            input_data={"metadata": call_data.get("metadata")},
            metadata={"segments_count": len(call_data.get("segments", []))}
        )
    
    prompt = build_qa_prompt(call_data)

    try:
        if client is None:
            error_result = {"error": "Thiếu cấu hình OPENAI_API_KEY/BASE_URL"}
            if trace:
                trace.update(output=error_result)
            return error_result
            
        if not MODEL_NAME:
            error_result = {"error": "Thiếu MODEL_NAME"}
            if trace:
                trace.update(output=error_result)
            return error_result

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

        messages = [
            {
                "role": "system",
                "content": "Bạn là chuyên gia phân tích chất lượng cuộc gọi. Đánh giá chính xác dựa trên dữ liệu âm học và transcript. Trả về JSON theo format đã cho."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"}
        )

        result_content = response.choices[0].message.content.strip()

        
        if result_content.startswith("```json"):
            result_content = result_content[7:]
        if result_content.endswith("```"):
            result_content = result_content[:-3]
        
        result_content = result_content.strip()
        result_json = json.loads(result_content)
        
        
        if trace:
            log_generation(
                trace=trace,
                name="llm_evaluation",
                model=MODEL_NAME,
                input_data=messages,
                output_data=result_json,
                metadata={
                    "temperature": 0,
                    "response_format": "json_object"
                },
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            )
            
            
            trace.update(
                output={
                    "status": "success",
                    "scores": {
                        "chao_xung_danh": result_json.get("chao_xung_danh"),
                        "ky_nang_noi": result_json.get("ky_nang_noi"),
                        "ky_nang_nghe": result_json.get("ky_nang_nghe"),
                        "thai_do": result_json.get("thai_do")
                    },
                    "muc_loi": result_json.get("muc_loi")
                }
            )

        return result_json

    except json.JSONDecodeError as json_err:
        print(f"Lỗi parse JSON: {json_err}")
        print(f"Nội dung từ model: {result_content}")
        error_result = {
            "error": "Lỗi phân tích JSON từ model",
            "raw_response": result_content
        }
        
        if trace:
            trace.update(output={"status": "error", "error": str(json_err)})
        
        return error_result
        
    except Exception as e:
        print(f"Lỗi khi gọi LLM API: {e}")
        error_result = {"error": str(e)}
        
        if trace:
            trace.update(output={"status": "error", "error": str(e)})
        
        return error_result
