"""
Repository pattern để thao tác với database
"""

from datetime import datetime
from typing import List, Optional, Dict, Any

from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from .models import Evaluation, Segment


class EvaluationRepository:

    @staticmethod
    def create(
        db: Session, task_id: str, filename: str, file_size_mb: float
    ) -> Evaluation:

        evaluation = Evaluation(
            task_id=task_id,
            filename=filename,
            file_size_mb=file_size_mb,
            status="pending",
        )
        db.add(evaluation)
        db.commit()
        db.refresh(evaluation)
        return evaluation

    @staticmethod
    def get_by_task_id(db: Session, task_id: str) -> Optional[Evaluation]:

        return db.query(Evaluation).filter(Evaluation.task_id == task_id).first()

    @staticmethod
    def get_by_id(db: Session, evaluation_id: int) -> Optional[Evaluation]:

        return db.query(Evaluation).filter(Evaluation.id == evaluation_id).first()

    @staticmethod
    def update_status(db: Session, task_id: str, status: str) -> Optional[Evaluation]:

        evaluation = EvaluationRepository.get_by_task_id(db, task_id)
        if evaluation:
            evaluation.status = status
            db.commit()
            db.refresh(evaluation)
        return evaluation

    @staticmethod
    def update_result(
        db: Session, task_id: str, result_data: Dict[str, Any]
    ) -> Optional[Evaluation]:

        evaluation = EvaluationRepository.get_by_task_id(db, task_id)
        if not evaluation:
            return None

        # Update các trường điểm
        evaluation.status = "completed"
        evaluation.completed_at = datetime.utcnow()
        evaluation.chao_xung_danh = result_data.get("chao_xung_danh")
        evaluation.ky_nang_noi = result_data.get("ky_nang_noi")
        evaluation.ky_nang_nghe = result_data.get("ky_nang_nghe")
        evaluation.thai_do = result_data.get("thai_do")
        evaluation.tong_diem = result_data.get("tong_diem")
        evaluation.muc_loi = result_data.get("muc_loi")
        evaluation.ly_do = result_data.get("ly_do")

        # Metadata
        metadata = result_data.get("metadata", {})
        evaluation.duration = metadata.get("duration")
        evaluation.turns = metadata.get("turns")
        evaluation.ratio_sales = metadata.get("ratio_sales")

        # Lưu raw result
        evaluation.raw_result = result_data

        db.commit()
        db.refresh(evaluation)
        return evaluation

    @staticmethod
    def update_error(
        db: Session, task_id: str, error_message: str
    ) -> Optional[Evaluation]:

        evaluation = EvaluationRepository.get_by_task_id(db, task_id)
        if evaluation:
            evaluation.status = "failed"
            evaluation.completed_at = datetime.utcnow()
            evaluation.error_message = error_message
            db.commit()
            db.refresh(evaluation)
        return evaluation

    @staticmethod
    def list_all(
        db: Session, limit: int = 50, offset: int = 0, status: Optional[str] = None
    ) -> List[Evaluation]:

        query = db.query(Evaluation)

        if status:
            query = query.filter(Evaluation.status == status)

        query = query.order_by(desc(Evaluation.created_at))
        query = query.limit(limit).offset(offset)

        return query.all()

    @staticmethod
    def count(db: Session, status: Optional[str] = None) -> int:

        query = db.query(func.count(Evaluation.id))
        if status:
            query = query.filter(Evaluation.status == status)
        return query.scalar()

    @staticmethod
    def get_statistics(db: Session) -> Dict[str, Any]:

        total = db.query(func.count(Evaluation.id)).scalar()
        pending = (
            db.query(func.count(Evaluation.id))
            .filter(Evaluation.status == "pending")
            .scalar()
        )
        processing = (
            db.query(func.count(Evaluation.id))
            .filter(Evaluation.status == "processing")
            .scalar()
        )
        completed = (
            db.query(func.count(Evaluation.id))
            .filter(Evaluation.status == "completed")
            .scalar()
        )
        failed = (
            db.query(func.count(Evaluation.id))
            .filter(Evaluation.status == "failed")
            .scalar()
        )

        avg_score = (
            db.query(func.avg(Evaluation.tong_diem))
            .filter(Evaluation.status == "completed", Evaluation.tong_diem.isnot(None))
            .scalar()
        )

        error_distribution = (
            db.query(Evaluation.muc_loi, func.count(Evaluation.id))
            .filter(Evaluation.status == "completed")
            .group_by(Evaluation.muc_loi)
            .all()
        )

        return {
            "total_evaluations": total,
            "pending": pending,
            "processing": processing,
            "completed": completed,
            "failed": failed,
            "average_score": round(float(avg_score), 2) if avg_score else None,
            "error_distribution": {level: count for level, count in error_distribution},
        }

    @staticmethod
    def delete(db: Session, task_id: str) -> bool:

        evaluation = EvaluationRepository.get_by_task_id(db, task_id)
        if evaluation:
            db.delete(evaluation)
            db.commit()
            return True
        return False


class SegmentRepository:
    """
    Repository cho operations liên quan đến Segment
    """

    @staticmethod
    def create_bulk(
        db: Session, evaluation_id: int, segments_data: List[Dict[str, Any]]
    ) -> List[Segment]:
        """
        Tạo nhiều segments cùng lúc
        """
        segments = []
        for seg_data in segments_data:
            segment = Segment(
                evaluation_id=evaluation_id,
                segment_number=seg_data.get("segment"),
                speaker=seg_data.get("speaker"),
                start_time=seg_data.get("start_time"),
                end_time=seg_data.get("end_time"),
                text=seg_data.get("text"),
                speed_spm=seg_data.get("speed_spm"),
                volume_db=seg_data.get("volume_db"),
                pitch_hz=seg_data.get("pitch_hz"),
                silence_ratio=seg_data.get("silence_ratio"),
            )
            segments.append(segment)

        db.add_all(segments)
        db.commit()
        return segments

    @staticmethod
    def get_by_evaluation_id(db: Session, evaluation_id: int) -> List[Segment]:
        """
        Lấy tất cả segments của một evaluation
        """
        return (
            db.query(Segment)
            .filter(Segment.evaluation_id == evaluation_id)
            .order_by(Segment.segment_number)
            .all()
        )
