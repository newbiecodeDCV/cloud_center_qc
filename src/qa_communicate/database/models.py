"""
Database models cho hệ thống QA Call Center
"""

from datetime import datetime

from sqlalchemy import Column, Integer, String, Float, Text, DateTime, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Evaluation(Base):
    """
    Bảng chính lưu kết quả đánh giá cuộc gọi
    """

    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(255), unique=True, nullable=False, index=True)

    filename = Column(String(500))
    file_size_mb = Column(Float)

    status = Column(String(50), nullable=False, default="pending", index=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    duration = Column(Float, nullable=True)
    turns = Column(Integer, nullable=True)
    ratio_sales = Column(Float, nullable=True)

    chao_xung_danh = Column(Integer, nullable=True)
    ky_nang_noi = Column(Integer, nullable=True)
    ky_nang_nghe = Column(Integer, nullable=True)
    thai_do = Column(Integer, nullable=True)

    tong_diem = Column(Float, nullable=True)
    muc_loi = Column(String(50), nullable=True)
    ly_do = Column(Text, nullable=True)

    error_message = Column(Text, nullable=True)

    raw_result = Column(JSON, nullable=True)

    segments = relationship(
        "Segment", back_populates="evaluation", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Evaluation(task_id='{self.task_id}', status='{self.status}', score={self.tong_diem})>"

    def to_dict(self):
        """Convert to dictionary for API response"""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "filename": self.filename,
            "file_size_mb": self.file_size_mb,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "duration": self.duration,
            "turns": self.turns,
            "ratio_sales": self.ratio_sales,
            "chao_xung_danh": self.chao_xung_danh,
            "ky_nang_noi": self.ky_nang_noi,
            "ky_nang_nghe": self.ky_nang_nghe,
            "thai_do": self.thai_do,
            "tong_diem": self.tong_diem,
            "muc_loi": self.muc_loi,
            "ly_do": self.ly_do,
            "error_message": self.error_message,
        }


class Segment(Base):
    """
    Bảng lưu các segment âm thanh của cuộc gọi
    """

    __tablename__ = "segments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    evaluation_id = Column(
        Integer,
        ForeignKey("evaluations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    segment_number = Column(Integer, nullable=False)  # Thứ tự segment (1, 2, 3...)
    speaker = Column(String(50), nullable=False)  # Sales | Customer

    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)

    text = Column(Text, nullable=True)

    speed_spm = Column(Float, nullable=True)
    volume_db = Column(Float, nullable=True)
    pitch_hz = Column(Float, nullable=True)
    silence_ratio = Column(Float, nullable=True)

    # Relationship
    evaluation = relationship("Evaluation", back_populates="segments")

    def __repr__(self):
        return f"<Segment(eval_id={self.evaluation_id}, segment={self.segment_number}, speaker='{self.speaker}')>"

    def to_dict(self):
        """Convert to dictionary"""
        return {
            "segment": self.segment_number,
            "speaker": self.speaker,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
            "speed_spm": self.speed_spm,
            "volume_db": self.volume_db,
            "pitch_hz": self.pitch_hz,
            "silence_ratio": self.silence_ratio,
        }


class DemoSession(Base):
    __tablename__ = "demo_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    notes = Column(Text, nullable=True)

    def __repr__(self):
        return f"<DemoSession(name='{self.session_name}', date={self.created_at})>"
