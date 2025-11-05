"""
Database connection và session management
"""

from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from .models import Base

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATABASE_DIR = PROJECT_ROOT / "data"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_PATH = DATABASE_DIR / "qa_database.db"

DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)


@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """
    Khởi tạo database - tạo tất cả tables
    """
    print(f"🔧 Initializing database at: {DATABASE_PATH}")
    Base.metadata.create_all(bind=engine)
    print("✅ Database initialized successfully!")


def drop_db():
    """
    Xóa tất cả tables (cẩn thận!)
    """
    print("⚠️  Dropping all tables...")
    Base.metadata.drop_all(bind=engine)
    print("✅ All tables dropped!")


@contextmanager
def get_db() -> Session:
    """
    Context manager để lấy database session

    Usage:
        with get_db() as db:
            result = db.query(Evaluation).all()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db_session() -> Session:
    """
    Lấy database session (cho FastAPI dependency injection)

    Usage trong FastAPI:
        def my_endpoint(db: Session = Depends(get_db_session)):
            ...
    """
    return SessionLocal()
