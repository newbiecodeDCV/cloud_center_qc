# -*- coding: utf-8 -*-
import gradio as gr
import os
from pydub import AudioSegment
from src.qa_communicate.audio_processing.qa import call_qa_api
from src.qa_communicate.core.utils import create_task_id
from fastapi import FastAPI
import fastapi
import argparse
import uvicorn
import requests
import tempfile


app = FastAPI()


def get_root_url(request: fastapi.Request, route_path: str, root_path) -> str:
    return "https://speech.aiservice.vn/asr/cloud_qa_demo"


def download_audio_from_url(url: str):
    try:
        r = requests.get(url, allow_redirects=True)
        if r.status_code != 200:
            return f"Lỗi tải file (status {r.status_code})"
        # Tạo file tạm và đoán phần mở rộng từ URL
        ext = url.split('.')[-1] if '.' in url else 'wav'
        temp_file = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False)
        temp_file.write(r.content)
        temp_file.close()
        # (Tùy chọn) Convert sang WAV nếu bạn cần đầu vào chuẩn
        audio = AudioSegment.from_file(temp_file.name)
        wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio.export(wav_file.name, format="wav")
        return wav_file.name
    except Exception as e:
        return f"Lỗi: {e}"


async def process_audio_and_evaluate(audio_file_path, audio_url, progress=gr.Progress()):
    """Xử lý audio qua API"""
    if audio_file_path is None and audio_url:
        audio_file_path = download_audio_from_url(audio_url)
    report_str = "Đang xử lý..."
    if not audio_file_path or not os.path.exists(audio_file_path):
        return "❌ Vui lòng tải lên một file âm thanh hợp lệ."
    progress(0.1, desc="📁 Đang đọc file audio...")
    try:
        with open(audio_file_path, 'rb') as f:
            audio_bytes = f.read()
    except Exception as e:
        return f"❌ Lỗi đọc file: {str(e)}"

    task_id = create_task_id(audio_bytes)
    progress(0.3, desc="🔄 Đang gửi yêu cầu đến API...")

    try:
        result = await call_qa_api(
            audio_bytes=audio_bytes,
            task_id=task_id,
            max_poll_seconds=180.0,
            poll_interval_seconds=2.0,
            verbose=True
        )
    except Exception as e:
        return f"❌ Lỗi khi gọi API: {str(e)}"

    progress(0.8, desc="📊 Đang xử lý kết quả...")
    if result.get('status') != 1:
        error_msg = result.get('message', 'Không xác định')
        return f"❌ Lỗi từ API: {error_msg}"
    dialogue_report = result.get('result', '')
    task_id = result.get('task_id', '')
    report_lines = []
    report_lines.append("╔════════════════════════════════════════════════════════════════╗")
    report_lines.append(f"║              📊 BÁO CÁO ĐÁNH GIÁ.ID cuộc gọi: {task_id}       ║")
    report_lines.append("╚════════════════════════════════════════════════════════════════╝")
    report_lines.append("")
    if dialogue_report:
        if isinstance(dialogue_report, str):
            report_lines.append(dialogue_report)
        else:
            report_lines.append(str(dialogue_report))
    else:
        report_lines.append("⚠️ API trả về thành công nhưng không có báo cáo.")
    report_lines.append("")
    report_lines.append("═══════════════════════════════════════════════════════════════")
    report_lines.append("✅ Hoàn tất!")
    report_str = "\n".join(report_lines)
    progress(1.0, desc="✅ Hoàn thành!")
    return report_str

custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}
.report-box textarea {
    font-family: 'Courier New', monospace !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
}
.main-header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 30px;
}
.analyze-button {
    width: 100% !important;
    height: 60px !important;
    font-size: 18px !important;
    font-weight: bold !important;
}
.info-box {
    background: #f0f7ff;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #667eea;
    margin-top: 15px;
}
"""

with gr.Blocks(title="Demo đánh giá chất lượng cuộc gọi", theme=gr.themes.Soft(), css=custom_css) as demo:
    # Header
    with gr.Row(elem_classes="main-header"):
        gr.Markdown("""
        #  Demo đánh giá chất lượng cuộc gọi 
        """)
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## 📤 Bước 1: Tải lên file audio")
            audio_input = gr.Audio(
                label="🎙️ Tải audio từ máy tính (.wav, .mp3, .m4a)",
                type="filepath",
                elem_classes="audio-input"
            )
            audio_url = gr.Textbox(label="Hoặc nhập URL audio")
            analyze_btn = gr.Button(
                "🚀 Bắt đầu xử lý",
                variant="primary",
                size="lg",
                elem_classes="analyze-button"
            )
            with gr.Group(elem_classes="info-box"):
                gr.Markdown("""### 📋 Hướng dẫn sử dụng:
                
                1. 📁 **Tải file**: Chọn file audio từ máy tính
                2. ▶️ **Bắt đầu**: Nhấn nút "Bắt đầu Xử lý"
                3. ⏳ **Chờ đợi**: Quá trình xử lý 1-2 phút
                4. ✅ **Kết quả**: Xem báo cáo bên phải
		5. **Những chức năng đã có**
		- [x] Chấm điểm kỹ năng giao tiếp  
		- [x] Chấm điểm kỹ năng bán hàng  
		6. **Những chức năng đang phát triển**
		- [ ] Đánh giá nhập liệu CRM  
		- [ ] Đánh giá mức lỗi  
		- [ ] Chấm điểm với thông tin trong tài liệu sản phẩm
                7. **Lưu ý**: trong quá trình test các chị note lại giúp em ID cuộc gọi được ghi
		ở đầu báo cáo để sau này bọn em dễ đối chiếu và cải thiện kết quả. Em cảm ơn các chị nhiều !
"""
)
        with gr.Column(scale=3):
            gr.Markdown("## 📊 Kết quả đánh giá")
            report_output = gr.Textbox(
                label="📄 Báo cáo Chi tiết",
                lines=25,
                max_lines=40,
                interactive=False,
                show_copy_button=True,
                placeholder="🔄 Kết quả xử lý sẽ hiển thị tại đây...\n\n"
                           "Sau khi tải file và nhấn 'Bắt đầu Xử lý',\n"
                           "hệ thống sẽ:\n\n"
                           "• Gửi audio đến API\n"
                           "• Poll kết quả định kỳ\n"
                           "• Hiển thị thông tin chi tiết\n\n"
                           "Vui lòng đợi trong giây lát...",
                elem_classes="report-box"
            )
    # Kết nối events
    analyze_btn.click(
        fn=process_audio_and_evaluate,
        inputs=[audio_input, audio_url],
        outputs=[report_output]
    )
    # Footer
    gr.Markdown("""
    ---
    <div style="text-align: center; color: #666; font-size: 13px; padding: 20px;">
        <p><b>🔧 Powered by Admicro AI Speech Team</b></p>
    </div>
    """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Processing QA System")
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    args = parser.parse_args()

    app = gr.mount_gradio_app(app, demo, path="/")
    uvicorn.run(app, host=args.server_name, port=args.server_port)
