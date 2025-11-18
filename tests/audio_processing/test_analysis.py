import pytest
import numpy as np
from unittest.mock import Mock
from audio_analysis import (
    AcousticAnalyzer,
    AudioSegment,
)  # Thay "your_module" bằng tên file thực tế


# --- Fixtures ---
@pytest.fixture
def mock_audio_data():
    """Audio giả lập: 10 giây @ 16kHz"""
    return np.random.randn(160000).astype(np.float32)


@pytest.fixture
def mock_non_silent_intervals():
    """Giả lập khoảng không im lặng: toàn bộ audio đều có tiếng"""
    return np.array([[0, 160000]])


@pytest.fixture
def acoustic_analyzer(mock_audio_data, mock_non_silent_intervals):
    return AcousticAnalyzer(mock_audio_data, 16000, mock_non_silent_intervals)


# --- Test _calculate_spm ---
class TestCalculateSPM:

    def test_empty_text_returns_zero(self, acoustic_analyzer):
        segment = Mock()
        segment.duration = 5.0
        segment.text = ""
        assert acoustic_analyzer._calculate_spm(segment) == 0.0

    def test_text_with_only_punctuation_returns_zero(self, acoustic_analyzer):
        segment = Mock()
        segment.duration = 2.0
        segment.text = "!!! ??? ... ,,"
        assert acoustic_analyzer._calculate_spm(segment) == 0.0

    def test_short_duration_returns_zero(self, acoustic_analyzer):
        segment = Mock()
        segment.duration = 0.25  # < 0.3
        segment.text = "Xin chào"
        assert acoustic_analyzer._calculate_spm(segment) == 0.0

    def test_normal_text_without_fillers(self, acoustic_analyzer):
        segment = Mock()
        segment.duration = 3.0
        segment.text = "Tài liệu hướng dẫn sử dụng"
        # "Tài", "liệu", "hướng", "dẫn", "sử", "dụng" → 6 âm tiết, all len > 1, no filler
        expected_spm = (6 / 3.0) * 60  # = 120.0
        assert acoustic_analyzer._calculate_spm(segment) == 120.0

    def test_text_with_fillers_and_short_words_removed(self, acoustic_analyzer):
        segment = Mock()
        segment.duration = 4.0
        segment.text = "À dạ vâng ạ xin chào bạn khỏe không ư a"
        # Sau lọc: chỉ giữ "xin", "chào", "bạn", "khỏe", "không" → 5 âm tiết
        expected_spm = (5 / 4.0) * 60  # = 75.0
        assert acoustic_analyzer._calculate_spm(segment) == 75.0

    def test_all_words_are_fillers_or_short(self, acoustic_analyzer):
        segment = Mock()
        segment.duration = 2.5
        segment.text = "À ư ơ dạ vâng ạ thì là"
        # Tất cả đều bị lọc → content_syllables = []
        assert acoustic_analyzer._calculate_spm(segment) == 0.0

    def test_mixed_case_and_punctuation(self, acoustic_analyzer):
        segment = Mock()
        segment.duration = 5.0
        segment.text = "Xin Chào, Bạn! Có khỏe không??"
        # → ["xin", "chào", "bạn", "có", "khỏe", "không"] → 6 âm tiết
        expected_spm = (6 / 5.0) * 60  # = 72.0
        assert acoustic_analyzer._calculate_spm(segment) == 72.0


# --- Test is_corrupted (trong AudioSegment) ---
class TestAudioSegmentCorruption:

    def test_short_segment_with_many_words_is_corrupted(self):
        seg_data = {
            "start": 10.0,
            "end": 10.2,  # duration = 0.2s < 0.25
            "speaker": "spk1",
            "text": "từ1 từ2 từ3 từ4",  # 4 words > MAX_WORDS_IN_SHORT_SEGMENT (3)
        }
        segment = AudioSegment(seg_data, sales_speaker_id="spk1")
        assert segment.is_corrupted() is True

    def test_short_segment_with_few_words_is_not_corrupted(self):
        seg_data = {
            "start": 10.0,
            "end": 10.2,
            "speaker": "spk1",
            "text": "xin chào",  # 2 words <= 3
        }
        segment = AudioSegment(seg_data, sales_speaker_id="spk1")
        assert segment.is_corrupted() is False

    def test_long_segment_is_never_corrupted(self):
        seg_data = {
            "start": 10.0,
            "end": 15.0,  # duration = 5s
            "speaker": "spk1",
            "text": "a " * 100,  # rất nhiều từ
        }
        segment = AudioSegment(seg_data, sales_speaker_id="spk1")
        assert segment.is_corrupted() is False


# --- Optional: Test integration with analyze_segment ---
def test_analyze_segment_returns_correct_keys(acoustic_analyzer):
    segment = Mock()
    segment.is_corrupted.return_value = False
    segment.duration = 2.0
    segment.text = "Kiểm tra tích hợp"
    segment.start_time = 0.0
    segment.end_time = 2.0

    result = acoustic_analyzer.analyze_segment(segment)
    expected_keys = {"speed_spm", "volume_db", "pitch_hz", "silence_ratio"}
    assert set(result.keys()) == expected_keys
    assert isinstance(result["speed_spm"], float)
