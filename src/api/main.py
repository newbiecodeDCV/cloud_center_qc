from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import sys
from pathlib import Path
import json
from datetime import datetime
import logging
import uuid


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.qa_communicate.audio_processing.analysis import extract_features
    from src.qa_communicate.evaluation.evaluator import get_qa_evaluation
    logger.info("Import thành công các module từ src/")
except ImportError as e:
    logger.error(f"Lỗi import: {e}")
    raise ImportError("Không tìm thấy module trong src/.")

# Khởi tạo FastAPI
app = FastAPI(
    title="Call Center QA API",
    description="API đánh giá chất lượng cuộc gọi Sales",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production nên giới hạn domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


RESULTS_DIR = Path("results") / "score"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


task_storage: Dict[str, Dict[str, Any]] = {}


class EvaluationResponse(BaseModel):
    """Response model cho kết quả đánh giá"""
    task_id: str = Field(..., description="ID tác vụ để tracking")
    status: str = Field(..., description="pending | processing | completed | failed")
    chao_xung_danh: Optional[int] = Field(None, ge=0, le=1, description="Điểm chào/xưng danh (0 hoặc 1)")
    ky_nang_noi: Optional[int] = Field(None, ge=0, le=1, description="Điểm kỹ năng nói (0 hoặc 1)")
    ky_nang_nghe: Optional[int] = Field(None, ge=0, le=1, description="Điểm kỹ năng nghe (0 hoặc 1)")
    thai_do: Optional[int] = Field(None, ge=0, le=1, description="Điểm thái độ (0 hoặc 1)")
    tong_diem: Optional[float] = Field(None, ge=0, le=2, description="Tổng điểm (0-2)")
    muc_loi: Optional[str] = Field(None, description="Mức lỗi: Không | M1 | M2 | M3")
    ly_do: Optional[str] = Field(None, description="Lý do chi tiết")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata cuộc gọi")
    error_message: Optional[str] = Field(None, description="Thông báo lỗi nếu có")
    created_at: Optional[str] = Field(None, description="Thời gian tạo task")
    completed_at: Optional[str] = Field(None, description="Thời gian hoàn thành")
    segments: Optional[List[Dict[str, Any]]] = Field(None, description="Dữ liệu các đoạn âm thanh")
    

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "completed",
                "chao_xung_danh": 1,
                "ky_nang_noi": 1,
                "ky_nang_nghe": 0,
                "thai_do": 1,
                "tong_diem": 1.2,
                "muc_loi": "M1",
                "ly_do": "- Chào hỏi tốt\n- Nói hơi nhanh\n- Chưa lắng nghe tốt",
                "metadata": {
                    "duration": 120.5,
                    "turns": 15,
                    "ratio_sales": 0.65
                }
            }
        }


class TaskStatusResponse(BaseModel):
    """Response cho trạng thái task"""
    task_id: str
    status: str
    message: str
    progress: Optional[float] = Field(None, ge=0, le=1, description="Tiến độ 0-1")




def validate_audio_file(filename: str) -> tuple[bool, str]:
    """Validate audio file format"""
    allowed_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
    file_ext = Path(filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        return False, f"Định dạng file không được hỗ trợ. Chỉ chấp nhận: {', '.join(allowed_extensions)}"
    
    return True, "OK"


def create_task_id() -> str:
    return str(uuid.uuid4())


def save_result_to_file(task_id: str, result: Dict[str, Any]) -> Path:
    """Lưu kết quả vào file JSON"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"evaluation_{task_id[:8]}_{timestamp}.json"
    output_path = RESULTS_DIR / output_filename
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        logger.info(f" Đã lưu kết quả vào: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f" Lỗi lưu file: {e}")
        raise


async def process_evaluation_task(task_id: str, audio_bytes: bytes):
    """
    Background task để xử lý đánh giá cuộc gọi
    """
    try:
        # Cập nhật trạng thái
        task_storage[task_id]["status"] = "processing"
        task_storage[task_id]["progress"] = 0.1
        logger.info(f"[{task_id}] Bắt đầu xử lý...")
        
        #  Phân tích âm học
        logger.info(f"[{task_id}] Đang phân tích acoustic features...")
        task_storage[task_id]["progress"] = 0.3
        
        analysis_result = await extract_features(audio_bytes)
        
        if analysis_result.get('status') != 1:
            error_msg = analysis_result.get('message', 'Lỗi không xác định')
            logger.error(f"[{task_id}] Lỗi phân tích: {error_msg}")
            task_storage[task_id].update({
                "status": "failed",
                "error_message": f"Lỗi phân tích audio: {error_msg}",
                "completed_at": datetime.now().isoformat()
            })
            return
        
        logger.info(f"[{task_id}] ✓ Phân tích acoustic thành công")
        task_storage[task_id]["progress"] = 0.6
        
        #  Đánh giá bằng LLM
        logger.info(f"[{task_id}] Đang chấm điểm bằng LLM...")
        
        data_for_llm = {
            'metadata': analysis_result.get('metadata'),
            'segments': analysis_result.get('segments')
        }
        
        evaluation_result = await get_qa_evaluation(data_for_llm)
        
        if not evaluation_result or "error" in evaluation_result:
            error_msg = evaluation_result.get('error', 'Lỗi không xác định') if evaluation_result else 'Không nhận được response'
            logger.error(f"[{task_id}] Lỗi LLM: {error_msg}")
            task_storage[task_id].update({
                "status": "failed",
                "error_message": f"Lỗi chấm điểm LLM: {error_msg}",
                "completed_at": datetime.now().isoformat()
            })
            return
        
        logger.info(f"[{task_id}] ✓ Chấm điểm thành công")
        task_storage[task_id]["progress"] = 0.9
        
        #  Xử lý kết quả 
        chao_xung_danh = int(evaluation_result.get('chao_xung_danh', 0))
        ky_nang_noi = int(evaluation_result.get('ky_nang_noi', 0))  
        ky_nang_nghe = int(evaluation_result.get('ky_nang_nghe', 0))  
        thai_do = int(evaluation_result.get('thai_do', 0)) 
        
        # Tính tổng điểm
        tong_diem = 0.2*(chao_xung_danh + ky_nang_noi) + 0.8 * (ky_nang_nghe + thai_do)
        
        
        muc_loi = str(evaluation_result.get('muc_loi', 'Không'))
        ly_do = str(evaluation_result.get('ly_do', 'Không có lý do chi tiết'))
        
        
        final_result = {
            "task_id": task_id,
            "status": "completed",
            "chao_xung_danh": chao_xung_danh,
            "ky_nang_noi": ky_nang_noi,
            "ky_nang_nghe": ky_nang_nghe,
            "thai_do": thai_do,
            "tong_diem": tong_diem,
            "muc_loi": muc_loi,
            "ly_do": ly_do,
            "metadata": analysis_result.get('metadata'),
            "created_at": task_storage[task_id]["created_at"],
            "completed_at": datetime.now().isoformat(),
            "segments": analysis_result.get('segments')
        }
        
        
        task_storage[task_id].update(final_result)
        task_storage[task_id]["progress"] = 1.0
        
        
        try:
            save_result_to_file(task_id, final_result)
        except Exception as e:
            logger.warning(f"[{task_id}] Không thể lưu file (nhưng vẫn có kết quả): {e}")
        
        logger.info(f"[{task_id}] ✓ Hoàn thành đánh giá. Tổng điểm: {tong_diem}/2")
        
    except Exception as e:
        logger.error(f"[{task_id}] ✗ Lỗi hệ thống: {e}", exc_info=True)
        task_storage[task_id].update({
            "status": "failed",
            "error_message": f"Lỗi hệ thống: {str(e)}",
            "completed_at": datetime.now().isoformat()
        })




@app.get("/")
async def root():
    return {
        "service": "Call Center QA API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_tasks": len([t for t in task_storage.values() if t["status"] == "processing"])
    }


@app.post("/api/v1/evaluate", response_model=TaskStatusResponse)
async def evaluate(
    audio_file: UploadFile = File(..., description="File âm thanh cuộc gọi"),
    background_tasks: BackgroundTasks = None
):
    """
    Đánh giá kỹ năng giao tiếp từ file audio cuộc gọi.
    
    Endpoint này xử lý bất đồng bộ:
    1. Trả về task_id ngay lập tức
    2. Xử lý audio trong background
    3. Client dùng task_id để kiểm tra kết quả tại /api/v1/task/{task_id}
    
    """
    is_valid, message = validate_audio_file(audio_file.filename)
    if not is_valid:
        logger.warning(f"File không hợp lệ: {audio_file.filename}")
        raise HTTPException(status_code=400, detail=message)
    
    # Tạo task ID
    task_id = create_task_id()
    logger.info(f"[{task_id}] Nhận request đánh giá file: {audio_file.filename}")
    
    try:
        # Đọc nội dung file
        audio_bytes = await audio_file.read()
        file_size_mb = len(audio_bytes) / (1024 * 1024)
        
        # Validate file size (max 50MB)
        if file_size_mb > 50:
            raise HTTPException(
                status_code=400,
                detail=f"File quá lớn ({file_size_mb:.2f}MB). Giới hạn: 50MB"
            )
        
        logger.info(f"[{task_id}] Đã đọc {file_size_mb:.2f}MB")
        
    
        task_storage[task_id] = {
            "task_id": task_id,
            "status": "pending",
            "filename": audio_file.filename,
            "file_size_mb": round(file_size_mb, 2),
            "created_at": datetime.now().isoformat(),
            "progress": 0.0
        }
        
        background_tasks.add_task(process_evaluation_task, task_id, audio_bytes)
        
        return TaskStatusResponse(
            task_id=task_id,
            status="pending",
            message=f"Đã nhận file '{audio_file.filename}'. Đang xử lý...",
            progress=0.0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{task_id}] Lỗi xử lý upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")


@app.get("/api/v1/task/{task_id}", response_model=EvaluationResponse)
async def get_task_result(task_id: str):
    """
    Lấy kết quả đánh giá theo task_id.
    
    **Status codes:**
    - `pending`: Đang chờ xử lý
    - `processing`: Đang xử lý
    - `completed`: Đã hoàn thành
    - `failed`: Xử lý thất bại
    """
    if task_id not in task_storage:
        logger.warning(f"Task ID không tồn tại: {task_id}")
        raise HTTPException(status_code=404, detail=f"Không tìm thấy task_id: {task_id}")
    
    task_data = task_storage[task_id]
    logger.info(f"[{task_id}] Truy vấn kết quả. Status: {task_data['status']}")
    
    return EvaluationResponse(**task_data)


@app.get("/api/v1/tasks")
async def list_tasks(
    status: Optional[str] = None,
    limit: int = 50
):
    """
    Liệt kê các task đánh giá.
    
    **Params:**
    - status: Lọc theo trạng thái (pending, processing, completed, failed)
    - limit: Số lượng task tối đa trả về (mặc định: 50)
    """
    tasks = list(task_storage.values())
    
    # Filter by status
    if status:
        tasks = [t for t in tasks if t["status"] == status]
    
    # Sort by created_at descending
    tasks.sort(key=lambda x: x["created_at"], reverse=True)
    
    # Limit
    tasks = tasks[:limit]
    
    return {
        "total": len(tasks),
        "tasks": tasks
    }


@app.delete("/api/v1/task/{task_id}")
async def delete_task(task_id: str):
    """Xóa task khỏi storage (chỉ xóa in-memory, không xóa file kết quả)"""
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy task_id: {task_id}")
    
    task_data = task_storage.pop(task_id)
    logger.info(f"[{task_id}] Đã xóa task")
    
    return {
        "message": "Đã xóa task thành công",
        "task_id": task_id,
        "status": task_data["status"]
    }


@app.get("/api/v1/statistics")
async def get_statistics():
    """Lấy thống kê tổng quan"""
    all_tasks = list(task_storage.values())
    
    stats = {
        "total_tasks": len(all_tasks),
        "pending": len([t for t in all_tasks if t["status"] == "pending"]),
        "processing": len([t for t in all_tasks if t["status"] == "processing"]),
        "completed": len([t for t in all_tasks if t["status"] == "completed"]),
        "failed": len([t for t in all_tasks if t["status"] == "failed"]),
    }
    
    completed_tasks = [t for t in all_tasks if t["status"] == "completed"]
    if completed_tasks:
        avg_score = sum(t.get("tong_diem", 0) for t in completed_tasks) / len(completed_tasks)
        stats["average_score"] = round(avg_score, 2)
    else:
        stats["average_score"] = None
    
    return stats



@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catch-all exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# ==================== STARTUP & SHUTDOWN ====================

@app.on_event("startup")
async def startup_event():
    """Actions on startup"""
    logger.info("="*60)
    logger.info("🚀 Call Center QA API đang khởi động...")
    logger.info(f"📁 Results directory: {RESULTS_DIR}")
    logger.info(f"📖 API Docs: http://localhost:8000/docs")
    logger.info("="*60)


@app.on_event("shutdown")
async def shutdown_event():
    """Actions on shutdown"""
    logger.info("🛑 API đang tắt...")
    
    # Lưu task storage vào file trước khi tắt (optional)
    try:
        backup_file = RESULTS_DIR / "task_storage_backup.json"
        with open(backup_file, 'w') as f:
            json.dump(task_storage, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Đã backup task storage vào {backup_file}")
    except Exception as e:
        logger.error(f"✗ Không thể backup task storage: {e}")


# ==================== RUN ====================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Đang khởi động API server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )