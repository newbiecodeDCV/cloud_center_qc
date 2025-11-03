"""
Test script để kiểm tra Langfuse integration
"""
import asyncio
import sys
from pathlib import Path
import os
# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.qa_communicate.core.langfuse_config import (
    LANGFUSE_ENABLED,
    create_trace,
    log_generation,
    log_span,
    flush_langfuse,
    langfuse_client
)


def test_basic_trace():
    """Test tạo trace cơ bản"""
    print("\n" + "="*60)
    print("TEST 1: Tạo trace cơ bản")
    print("="*60)
    
    if not LANGFUSE_ENABLED:
        print("⚠️  Langfuse không được bật. Kiểm tra file .env")
        return
    
    trace = create_trace(
        name="test_basic_trace",
        metadata={
            "test_type": "basic",
            "environment": "development"
        }
    )
    
    if trace:
        print("✅ Trace được tạo thành công")
        print(f"   Trace ID: {trace.id}")
        
        # Update trace
        trace.update(
            output={"status": "success", "message": "Test completed"}
        )
        print("✅ Trace được update thành công")
    else:
        print("❌ Không thể tạo trace")


def test_span_logging():
    """Test logging span"""
    print("\n" + "="*60)
    print("TEST 2: Logging span")
    print("="*60)
    
    if not LANGFUSE_ENABLED:
        print("⚠️  Langfuse không được bật")
        return
    
    trace = create_trace(
        name="test_span_logging",
        metadata={"test_type": "span"}
    )
    
    if trace:
        # Log span 1
        log_span(
            trace=trace,
            name="preprocessing",
            input_data={"raw_text": "Hello world"},
            output_data={"processed_text": "hello world"},
            metadata={"step": "1/3"}
        )
        print("✅ Span 1 logged")
        
        # Log span 2
        log_span(
            trace=trace,
            name="analysis",
            input_data={"processed_text": "hello world"},
            output_data={"analysis": "positive"},
            metadata={"step": "2/3"}
        )
        print("✅ Span 2 logged")
        
        trace.update(output={"status": "success"})
    else:
        print("❌ Không thể tạo trace")


def test_generation_logging():
    """Test logging LLM generation"""
    print("\n" + "="*60)
    print("TEST 3: Logging LLM generation")
    print("="*60)
    
    if not LANGFUSE_ENABLED:
        print("⚠️  Langfuse không được bật")
        return
    
    trace = create_trace(
        name="test_llm_generation",
        metadata={"test_type": "generation"}
    )
    
    if trace:
        # Giả lập LLM call
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What is AI?"}
        ]
        
        output = {
            "response": "AI stands for Artificial Intelligence..."
        }
        
        log_generation(
            trace=trace,
            name="gpt4_call",
            model="gpt-4.1-mini",
            input_data=messages,
            output_data=output,
            metadata={
                "temperature": 0.7,
                "max_tokens": 100
            },
            usage={
                "prompt_tokens": 20,
                "completion_tokens": 50,
                "total_tokens": 70
            }
        )
        print("✅ Generation logged")
        
        trace.update(output={"status": "success", "tokens_used": 70})
    else:
        print("❌ Không thể tạo trace")


def test_nested_traces():
    """Test nested traces (trace trong trace)"""
    print("\n" + "="*60)
    print("TEST 4: Nested traces")
    print("="*60)
    
    if not LANGFUSE_ENABLED:
        print("⚠️  Langfuse không được bật")
        return
    
    # Main trace
    main_trace = create_trace(
        name="main_evaluation_pipeline",
        metadata={"test_type": "nested"}
    )
    
    if main_trace:
        # Sub-span 1: Audio processing
        log_span(
            trace=main_trace,
            name="audio_processing",
            input_data={"audio_file": "test.wav"},
            metadata={"step": "1/3", "duration_seconds": 2.5}
        )
        print("✅ Sub-span 1 logged")
        
        # Sub-span 2: Feature extraction
        log_span(
            trace=main_trace,
            name="feature_extraction",
            input_data={"processed_audio": "..."},
            output_data={"features": ["pitch", "volume", "speed"]},
            metadata={"step": "2/3"}
        )
        print("✅ Sub-span 2 logged")
        
        # Sub-span 3: LLM evaluation
        log_generation(
            trace=main_trace,
            name="llm_evaluation",
            model="gpt-4.1-mini",
            input_data={"features": ["pitch", "volume", "speed"]},
            output_data={"score": 0.85, "feedback": "Good communication"},
            metadata={"step": "3/3"},
            usage={"total_tokens": 150}
        )
        print("✅ Sub-span 3 (generation) logged")
        
        main_trace.update(
            output={
                "status": "success",
                "total_score": 0.85,
                "evaluation_time_seconds": 5.2
            }
        )
        print("✅ Main trace updated")
    else:
        print("❌ Không thể tạo main trace")


def test_error_handling():
    """Test error handling"""
    print("\n" + "="*60)
    print("TEST 5: Error handling")
    print("="*60)
    
    if not LANGFUSE_ENABLED:
        print("⚠️  Langfuse không được bật")
        return
    
    trace = create_trace(
        name="test_error_handling",
        metadata={"test_type": "error"}
    )
    
    if trace:
        # Giả lập lỗi
        log_span(
            trace=trace,
            name="failing_step",
            input_data={"data": "test"},
            metadata={"step": "1/2"}
        )
        
        trace.update(
            output={
                "status": "error",
                "error_type": "ValueError",
                "error_message": "Invalid input format"
            }
        )
        print("✅ Error được log thành công")
    else:
        print("❌ Không thể tạo trace")


def main():
    """Run all tests"""
    print("\n" + "🧪" * 30)
    print("BẮT ĐẦU TEST LANGFUSE INTEGRATION")
    print("🧪" * 30)
    
    # Check status
    print(f"\n📊 Langfuse Status:")
    print(f"   Enabled: {LANGFUSE_ENABLED}")
    if LANGFUSE_ENABLED and langfuse_client:
        print(f"   ✅ Client initialized successfully")
    else:
        print(f"   ❌ Client not initialized")
        print(f"\n💡 Hướng dẫn:")
        print(f"   1. Thêm LANGFUSE_ENABLED=true vào .env")
        print(f"   2. Thêm LANGFUSE_PUBLIC_KEY và LANGFUSE_SECRET_KEY")
        print(f"   3. (Optional) Thêm LANGFUSE_HOST nếu dùng self-hosted")
        return
    
    # Run tests
    test_basic_trace()
    test_span_logging()
    test_generation_logging()
    test_nested_traces()
    test_error_handling()
    
    # Add debug log trong test script
    print(f"Public Key: {os.getenv('LANGFUSE_PUBLIC_KEY')[:10]}...")
    print(f"Host: {os.getenv('LANGFUSE_HOST')}")
    
    # Flush events
    print("\n" + "="*60)
    print("FLUSHING EVENTS")
    print("="*60)
    flush_langfuse()
    
    # ✅ FIX: Wait for async events to be sent
    import time
    print("⏳ Waiting for events to be sent to server...")
    for i in range(5, 0, -1):
        print(f"   {i} seconds remaining...", end="\r")
        time.sleep(1)
    print("\n✅ Tất cả events đã được gửi lên Langfuse")
    
    print("\n" + "🎉" * 30)
    print("HOÀN THÀNH TẤT CẢ TESTS")
    print("🎉" * 30)
    print("\n💡 Kiểm tra kết quả tại Langfuse Dashboard:")
    print("   https://cloud.langfuse.com (hoặc self-hosted URL)")
    print("\n⚠️  Lưu ý:")
    print("   - Đợi 10-30 giây để dashboard refresh")
    print("   - Check time filter trong dashboard (expand to 1 hour)")
    print("   - Click vào 'Traces' tab để xem chi tiết\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test bị dừng bởi người dùng")
        flush_langfuse()
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Lỗi không mong đợi: {e}")
        flush_langfuse()
        sys.exit(1)