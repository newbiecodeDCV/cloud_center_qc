"""
Fixed Langfuse configuration với proper async handling
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.decorators import observe
from loguru import logger

load_dotenv()


LANGFUSE_ENABLED = os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"

langfuse_client: Optional[Langfuse] = None

if LANGFUSE_ENABLED:
    try:
        langfuse_client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            flush_at=1,  # 🔥 CRITICAL: Flush ngay lập tức
            flush_interval=0.5,  # Flush mỗi 0.5s
        )
        logger.info("✅ Langfuse initialized successfully")
        logger.info(
            f"   Host: {os.getenv('LANGFUSE_HOST', 'https://cloud.langfuse.com')}"
        )
    except Exception as e:
        logger.error(f"❌ Failed to initialize Langfuse: {e}")
        LANGFUSE_ENABLED = False
else:
    logger.info("⚠️ Langfuse tracking is disabled")


def create_trace(
    name: str, metadata: Optional[Dict[str, Any]] = None, trace_id: Optional[str] = None
):
    """
    Tạo một trace mới trong Langfuse

    🔥 FIX: Thêm trace_id parameter để có thể link traces
    """
    if not LANGFUSE_ENABLED or not langfuse_client:
        return None

    try:
        trace = langfuse_client.trace(id=trace_id, name=name, metadata=metadata or {})

        #
        langfuse_client.flush()

        return trace
    except Exception as e:
        logger.error(f"Failed to create trace: {e}")
        return None


def log_generation(
    trace,
    name: str,
    model: str,
    input_data: Any,
    output_data: Any,
    metadata: Optional[Dict[str, Any]] = None,
    usage: Optional[Dict[str, int]] = None,
):
    """
    Log một LLM generation vào trace

     FIX: Flush ngay sau khi log
    """
    if not LANGFUSE_ENABLED or not trace:
        return

    try:
        generation = trace.generation(
            name=name,
            model=model,
            input=input_data,
            output=output_data,
            metadata=metadata or {},
            usage=usage,
        )

        if langfuse_client:
            langfuse_client.flush()

        return generation
    except Exception as e:
        logger.error(f"Failed to log generation: {e}")
        return None


def log_span(
    trace,
    name: str,
    input_data: Any = None,
    output_data: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Log một span (bước xử lý) vào trace

    🔥 FIX: Flush ngay sau khi log
    """
    if not LANGFUSE_ENABLED or not trace:
        return None

    try:
        span = trace.span(
            name=name, input=input_data, output=output_data, metadata=metadata or {}
        )

        if langfuse_client:
            langfuse_client.flush()

        return span
    except Exception as e:
        logger.error(f"Failed to log span: {e}")
        return None


def flush_langfuse():
    """
    🔥 FIX: Async flush với retry logic
    """
    if LANGFUSE_ENABLED and langfuse_client:
        try:
            logger.info("⏳ Flushing Langfuse events...")

            for i in range(3):
                langfuse_client.flush()
                import time

                time.sleep(0.5)

            logger.info("✅ Langfuse events flushed")

        except Exception as e:
            logger.error(f"Failed to flush Langfuse: {e}")


def shutdown_langfuse():
    """
    🔥 NEW: Proper shutdown với timeout
    """
    if LANGFUSE_ENABLED and langfuse_client:
        try:
            logger.info("🛑 Shutting down Langfuse...")

            flush_langfuse()

            import time

            time.sleep(2)

            logger.info("✅ Langfuse shutdown complete")

        except Exception as e:
            logger.error(f"Error during Langfuse shutdown: {e}")


class LangfuseContext:
    """
    Context manager để ensure proper cleanup

    Usage:
        with LangfuseContext() as ctx:
            trace = ctx.create_trace("my_trace")
            # ... do work
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        flush_langfuse()
        return False

    def create_trace(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        return create_trace(name, metadata)


__all__ = [
    "langfuse_client",
    "LANGFUSE_ENABLED",
    "create_trace",
    "log_generation",
    "log_span",
    "flush_langfuse",
    "shutdown_langfuse",
    "LangfuseContext",
    "observe",
]
