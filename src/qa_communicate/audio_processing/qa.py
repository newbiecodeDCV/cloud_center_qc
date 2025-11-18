import asyncio
import json
from io import BytesIO

import httpx
from loguru import logger


PRIVATE_TOKEN = "41ad97aa3c747596a4378cc8ba101fe70beb3f5f70a75407a30e6ddab668310d"


async def call_qa_api(
    audio_bytes: bytes,
    task_id: int,
    max_poll_seconds: float = 60.0,
    poll_interval_seconds: float = 0.5,
    verbose: bool = False,
):
    """Gọi API hội thoại bất đồng bộ và chờ lấy kết quả.

    Thực hiện hai bước:
    - Upload audio (wav) để khởi tạo tác vụ.
    - Poll endpoint kết quả cho đến khi hoàn tất hoặc thất bại.

    Tham số:
    - audio_bytes: dữ liệu âm thanh dạng bytes (định dạng wav) để nhận dạng hội thoại.
    - task_id: mã tác vụ (số nguyên) để theo dõi tiến trình trên dịch vụ.

    Trả về:
    - dict với các khóa:
      - status: 1 nếu thành công, -1 nếu thất bại.
      - task_id: mã tác vụ từ dịch vụ (nếu có).
      - dialogue: danh sách đoạn hội thoại theo speaker (chỉ có khi thành công).
      - message: thông điệp mô tả trạng thái.

    """
    dialogue_url = "https://speech.aiservice.vn/asr/cloud_qa"
    result_url = "https://speech.aiservice.vn/asr/cloud_qa_result"
    headers = {"Authorization": PRIVATE_TOKEN}
    files = {"file": ("dialogue.wav", BytesIO(audio_bytes), "audio/wav")}
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Bước 1: Gửi audio để khởi tạo task
            response = await client.post(
                dialogue_url,
                headers=headers,
                data={"task_id": str(task_id)},
                files=files,
            )
            # Kiểm tra status code trước khi parse JSON
            if response.status_code != 200:
                error_msg = f"API returned status {response.status_code}"
                if response.text:
                    error_msg += f": {response.text[:200]}"  # Giới hạn độ dài log
                logger.error(error_msg)
                return {
                    "status": -1,
                    "message": f"Failed to start dialogue processing: HTTP {response.status_code}",
                }
            # Parse JSON response
            try:
                response_data = response.json()
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from response: {e}")
                logger.error(
                    f"Response content: {response.text[:500]}"
                )  # Log 500 ký tự đầu
                return {"status": -1, "message": "Invalid JSON response from server"}
            task_id = response_data.get("task_id", None)
            if not task_id:
                return {
                    "status": -1,
                    "message": "Failed to start dialogue processing: No task_id returned",
                }
            # Bước 2: Poll kết quả
            result_payload = {"task_id": task_id}
            elapsed = 0.0

            while elapsed < max_poll_seconds:
                try:
                    result_response = await client.post(
                        result_url, headers=headers, data=result_payload
                    )

                    # Kiểm tra status code cho result API
                    if result_response.status_code != 200:
                        if verbose:
                            print(
                                f"[polling] HTTP error: {result_response.status_code}"
                            )
                        await asyncio.sleep(poll_interval_seconds)
                        elapsed += poll_interval_seconds
                        continue

                    # Parse JSON cho result
                    try:
                        result_data = result_response.json()
                        print(result_data)
                    except json.JSONDecodeError:
                        if verbose:
                            print(
                                f"[polling] JSON parse error, response: {result_response.text[:100]}"
                            )
                        await asyncio.sleep(poll_interval_seconds)
                        elapsed += poll_interval_seconds
                        continue

                    status = result_data.get("status")
                    if status == 1:
                        return result_data
                    elif status == -1:
                        return {
                            "status": -1,
                            "task_id": task_id,
                            "message": "Get dialogue failed: API returned status -1",
                        }

                    if verbose:
                        print(
                            f"[polling] task_id={task_id}, elapsed={elapsed:.1f}s, status={status}"
                        )
                    await asyncio.sleep(poll_interval_seconds)
                    elapsed += poll_interval_seconds
                except Exception as e:
                    logger.error(f"Error during polling: {e}")
                    await asyncio.sleep(poll_interval_seconds)
                    elapsed += poll_interval_seconds
            # Timeout
            return {
                "status": -1,
                "task_id": task_id,
                "message": f"Timeout after {max_poll_seconds}s while waiting for dialogue result",
            }
        except httpx.TimeoutException:
            logger.error("Request timeout")
            return {"status": -1, "message": "Request timeout"}
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            return {"status": -1, "message": f"Request error: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {"status": -1, "message": f"Unexpected error: {str(e)}"}
