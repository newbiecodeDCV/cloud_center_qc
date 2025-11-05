import json
import os
from typing import Dict, Any
from loguru import logger
import openai
from dotenv import load_dotenv

from src.qa_communicate.core.langfuse_config import log_generation, LANGFUSE_ENABLED
from src.qa_communicate.prompt.prompts import build_qa_prompt


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
    logger.warning("⚠️ Thiếu OPENAI_API_KEY hoặc BASE_URL trong file .env")


async def get_qa_evaluation(
    call_data: Dict[str, Any], trace=None, parent_span=None
) -> Dict:
    """
    Evaluate communication skills using LLM with Langfuse tracing.

    Steps:
    1. Build prompt from call data
    2. Call LLM (OpenAI AsyncClient)
    3. Parse response
    """
    try:
        # Validate configuration
        if client is None:
            error_msg = "Thiếu cấu hình OPENAI_API_KEY/BASE_URL"
            logger.error(error_msg)
            return {"error": error_msg}

        if not MODEL_NAME:
            error_msg = "Thiếu MODEL_NAME trong .env"
            logger.error(error_msg)
            return {"error": error_msg}

        # Sub-step 1: Build prompt
        if parent_span:
            prompt_span = parent_span.span(
                name="build_prompt", input={"call_data_keys": list(call_data.keys())}
            )

        logger.info("Building evaluation prompt...")
        prompt = build_qa_prompt(call_data)

        if parent_span:
            prompt_span.end(output={"prompt": prompt})

        # Sub-step 2: Call LLM
        logger.info(
            f"Calling OpenAI API ({MODEL_NAME}) for communication evaluation..."
        )

        messages = [
            {
                "role": "system",
                "content": "Bạn là chuyên gia phân tích chất lượng cuộc gọi. Đánh giá chính xác dựa trên dữ liệu âm học và transcript. Trả về JSON theo format đã cho.",
            },
            {"role": "user", "content": prompt},
        ]

        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )

        result_content = response.choices[0].message.content.strip()

        # Log LLM generation to Langfuse
        if trace and LANGFUSE_ENABLED:
            usage_data = response.usage
            log_generation(
                trace=trace,
                name="llm_communication_evaluation",
                model=MODEL_NAME,
                input_data={
                    "messages": messages,
                    "metadata_summary": call_data.get("metadata", {}),
                },
                output_data=result_content,
                metadata={
                    "evaluation_type": "communication_skills",
                    "temperature": 0,
                    "response_format": "json_object",
                },
                usage={
                    "prompt_tokens": usage_data.prompt_tokens if usage_data else 0,
                    "completion_tokens": (
                        usage_data.completion_tokens if usage_data else 0
                    ),
                    "total_tokens": usage_data.total_tokens if usage_data else 0,
                },
            )

        # Sub-step 3: Parse response
        if parent_span:
            parse_span = parent_span.span(
                name="parse_llm_response", input={"response": result_content}
            )

        logger.info("Parsing LLM response...")

        # Xử lý markdown block (fallback nếu model vẫn trả về)
        if result_content.startswith("```json"):
            result_content = result_content[7:]
        if result_content.endswith("```"):
            result_content = result_content[:-3]

        result_content = result_content.strip()

        try:
            evaluation_result = json.loads(result_content)

            if parent_span:
                parse_span.end(
                    output={"status": "success", "result": evaluation_result}
                )

            logger.info("✓ Successfully parsed communication evaluation")
            return evaluation_result

        except json.JSONDecodeError as json_err:
            logger.error(f"JSON decode error: {json_err}")
            logger.error(f"Raw response: {result_content[:500]}")

            if parent_span:
                parse_span.end(
                    output={
                        "status": "error",
                        "error": str(json_err),
                        "raw_response_preview": result_content[:200],
                    }
                )

            return {
                "error": "Lỗi phân tích JSON từ model",
                "raw_response": result_content,
            }

    except Exception as e:
        logger.error(f"Error in get_qa_evaluation: {e}", exc_info=True)

        if parent_span:
            error_span = parent_span.span(
                name="error_handling",
                input={"error": str(e), "error_type": type(e).__name__},
            )
            error_span.end(output={"status": "failed"})

        return {"error": str(e), "error_type": type(e).__name__}
