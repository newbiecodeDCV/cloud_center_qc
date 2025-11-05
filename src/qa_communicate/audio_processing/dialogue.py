import asyncio
from io import BytesIO

import httpx

PRIVATE_TOKEN = "41ad97aa3c747596a4378cc8ba101fe70beb3f5f70a75407a30e6ddab668310d"


async def call_dialogue_api(
    audio_bytes: bytes,
    task_id: int,
    *,
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
    dialogue_url = "https://speech.aiservice.vn/asr/dialogue"
    result_url = "https://speech.aiservice.vn/asr/dialogue_result"
    headers = {"Authorization": PRIVATE_TOKEN}
    files = {"file": ("dialogue.wav", BytesIO(audio_bytes), "audio/wav")}
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            dialogue_url, headers=headers, data={"task_id": str(task_id)}, files=files
        )
        task_id = response.json().get("task_id", None)
        if not task_id:
            return {"status": -1, "message": "Failed to start dialogue processing"}
        result_payload = {"task_id": task_id}
        elapsed = 0.0
        while True:
            result_response = await client.post(
                result_url, headers=headers, data=result_payload
            )
            result_data = result_response.json()
            if result_data["status"] == 1:
                return {
                    "status": 1,
                    "task_id": task_id,
                    "dialogue": result_data["result"]["spk_dialogue"],
                    "message": "Get dialogue complete",
                }
            elif result_data["status"] == -1:
                return {
                    "status": -1,
                    "task_id": task_id,
                    "message": "Get dialogue failed",
                }
            if verbose:
                # In ra tiến trình polling
                print(
                    f"[polling] task_id={task_id}, elapsed={elapsed:.1f}s, status={result_data.get('status')}"
                )
            if elapsed >= max_poll_seconds:
                return {
                    "status": -1,
                    "task_id": task_id,
                    "message": f"Timeout after {max_poll_seconds}s while waiting for dialogue result",
                }
            await asyncio.sleep(poll_interval_seconds)
            elapsed += poll_interval_seconds
