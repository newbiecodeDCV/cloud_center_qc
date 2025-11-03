"""
Langfuse configuration and utilities for LLM observability
"""
import os
from typing import Optional, Dict, Any
from functools import wraps
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe
from logging import getLogger

load_dotenv()
logger = getLogger(__name__)


LANGFUSE_ENABLED = os.getenv("LANGFUSE_ENABLED", "false").lower() == "true"


langfuse_client: Optional[Langfuse] = None

if LANGFUSE_ENABLED:
    try:
        langfuse_client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST")
        )
        logger.info("✅ Langfuse initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Langfuse: {e}")
        LANGFUSE_ENABLED = False
else:
    logger.info("⚠️ Langfuse tracking is disabled")


def create_trace(name: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Tạo một trace mới trong Langfuse
    
    Args:
        name: Tên của trace 
        metadata: Metadata bổ sung 
    
    Returns:
        Trace object hoặc None nếu Langfuse không được bật
    """
    if not LANGFUSE_ENABLED or not langfuse_client:
        return None
    
    try:
        trace = langfuse_client.trace(
            name=name,
            metadata=metadata or {}
        )
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
    usage: Optional[Dict[str, int]] = None
):
    """
    Log một LLM generation vào trace
    
    Args:
        trace: Trace object từ create_trace()
        name: Tên generation (ví dụ: "evaluate_script", "classify_utterances")
        model: Tên model (ví dụ: "gpt-4.1-mini")
        input_data: Input prompt hoặc messages
        output_data: Output từ LLM
        metadata: Metadata bổ sung
        usage: Token usage (prompt_tokens, completion_tokens, total_tokens)
    """
    if not LANGFUSE_ENABLED or not trace:
        return
    
    try:
        trace.generation(
            name=name,
            model=model,
            input=input_data,
            output=output_data,
            metadata=metadata or {},
            usage=usage
        )
    except Exception as e:
        logger.error(f"Failed to log generation: {e}")


def log_span(
    trace,
    name: str,
    input_data: Any = None,
    output_data: Any = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Log một span (bước xử lý) vào trace
    
    Args:
        trace: Trace object
        name: Tên span 
        input_data: Input của span
        output_data: Output của span
        metadata: Metadata bổ sung
    """
    if not LANGFUSE_ENABLED or not trace:
        return
    
    try:
        span = trace.span(
            name=name,
            input=input_data,
            output=output_data,
            metadata=metadata or {}
        )
        return span
    except Exception as e:
        logger.error(f"Failed to log span: {e}")
        return None


def flush_langfuse():
    """
    Flush tất cả events sang Langfuse server
    (Gọi khi shutdown application)
    """
    if LANGFUSE_ENABLED and langfuse_client:
        try:
            langfuse_client.flush()
            logger.info("✅ Langfuse events flushed")
        except Exception as e:
            logger.error(f"Failed to flush Langfuse: {e}")


def tracked_async_function(func_name: str):
    """
    Decorator để tự động track async function
    
    Usage:
        @tracked_async_function("evaluate_communication")
        async def evaluate_communication(...):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not LANGFUSE_ENABLED:
                return await func(*args, **kwargs)
            
            
            trace = create_trace(
                name=func_name,
                metadata={
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                }
            )
            
            try:
            
                result = await func(*args, **kwargs)
                
                
                if trace:
                    trace.update(
                        output={"status": result.get("status") if isinstance(result, dict) else "unknown"}
                    )
                
                return result
            except Exception as e:
            
                if trace:
                    trace.update(
                        output={"error": str(e)}
                    )
                raise
        
        return wrapper
    return decorator



__all__ = [
    "langfuse_client",
    "LANGFUSE_ENABLED",
    "create_trace",
    "log_generation",
    "log_span",
    "flush_langfuse",
    "tracked_async_function",
    "observe"  ]