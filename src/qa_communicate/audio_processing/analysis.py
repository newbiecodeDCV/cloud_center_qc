import numpy as np
import librosa
import re
from typing import List, Dict, Tuple
from src.qa_communicate.audio_processing.dialogue import call_dialogue_api
from src.qa_communicate.core.utils import create_task_id
from src.qa_communicate.audio_processing.speaker_validator import (
    validate_and_fix_speakers,
)
from io import BytesIO
from underthesea import word_tokenize

# Ngưỡng để xác định một segment có thể bị lỗi từ API
MIN_DURATION_FOR_VALID_SEGMENT = 0.25
MAX_WORDS_IN_SHORT_SEGMENT = 3


class AudioSegment:
    """Đại diện cho một phân đoạn âm thanh trong cuộc hội thoại"""

    def __init__(self, segment_data: Dict, sales_speaker_id: str):
        self.start_time = float(segment_data.get("start", 0.0))
        self.end_time = float(segment_data.get("end", self.start_time))
        self.original_speaker_id = str(segment_data.get("speaker", "unknown"))
        self.speaker_label = (
            "Sales" if self.original_speaker_id == str(sales_speaker_id) else "Customer"
        )
        self.text = segment_data.get("text", "")
        self.duration = self.end_time - self.start_time
        self.word_count = len(self.text.split()) if self.text else 0

    def is_corrupted(self) -> bool:
        """Kiểm tra segment có bị lỗi từ API hay không"""
        return (
            self.duration < MIN_DURATION_FOR_VALID_SEGMENT
            and self.word_count > MAX_WORDS_IN_SHORT_SEGMENT
        )


class AcousticAnalyzer:
    """Phân tích các đặc điểm acoustic của segment"""

    # Từ đệm thực sự (filler words) - âm thanh do dự, không mang nghĩa
    FILLER_WORDS = {
        "ờ",
        "ừm",
        "ừ",
        "à",
        "ơ",
        "ê",
        "ư",
        "hử",
        "kiểu",
        "dạng",
        "ấy là",
        "thì là",
        "là là",
    }

    # Từ lịch sự KHÔNG phải filler - GIỮ LẠI khi tính SPM và disfluency
    POLITE_WORDS = {"dạ", "vâng", "ạ", "nhé", "nhá", "nha", "nhỉ", "ơi"}

    def __init__(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        non_silent_intervals: List[Tuple[int, int]],
    ):
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.non_silent_intervals = non_silent_intervals

    def analyze_segment(self, segment: AudioSegment) -> Dict:
        """Phân tích acoustic features cho một segment"""
        if segment.is_corrupted():
            return {
                "speed_spm": 0.0,
                "volume_db": 0.0,
                "pitch_hz": 0.0,
                "silence_ratio": 0.0,
                "filler_count": 0,
                "restart_count": 0,
                "disfluency_rate": 0.0,
            }

        speed_spm = self._calculate_spm(segment)
        volume_db = self._calculate_volume(segment)
        pitch_hz = self._calculate_pitch(segment)
        silence_ratio = self._calculate_silence_ratio(segment)
        disfluency = self._calculate_disfluency_metrics(segment)

        return {
            "speed_spm": float(speed_spm),
            "volume_db": float(volume_db),
            "pitch_hz": float(pitch_hz),
            "silence_ratio": float(silence_ratio),
            **disfluency,
        }

    def _calculate_spm(self, segment: AudioSegment) -> float:
        """Tính tốc độ nói (SPM - syllables per minute) cho tiếng Việt

        Lọc bỏ: từ đệm thực sự (ờ, ừm, à, kiểu...)
        GIỮ LẠI: từ lịch sự (dạ, vâng, ạ), từ có nghĩa
        """
        if segment.duration <= 0.3:
            return 0.0

        text = segment.text
        if not text or not text.strip():
            return 0.0

        # Chuẩn hóa: lowercase + giữ chữ, số, khoảng trắng
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        if not cleaned:
            return 0.0

        # Tokenize tiếng Việt bằng underthesea
        try:
            words = word_tokenize(cleaned, format="text").split()
        except:
            # Fallback nếu underthesea lỗi
            words = cleaned.split()

        # Lọc bỏ CHỈ từ đệm thực sự (ờ, ừm, à, kiểu...)
        # GIỮ LẠI từ lịch sự (dạ, vâng, ạ) và từ có nghĩa
        content_syllables = [
            w for w in words if len(w) > 1 and w not in self.FILLER_WORDS
        ]

        if content_syllables:
            spm = (len(content_syllables) / segment.duration) * 60.0
            return round(spm, 2)

        return 0.0

    def _calculate_volume(self, segment: AudioSegment) -> float:
        """Tính âm lượng (dB)"""
        start_sample = int(segment.start_time * self.sample_rate)
        end_sample = int(segment.end_time * self.sample_rate)
        segment_audio = self.audio_data[start_sample:end_sample]

        if len(segment_audio) > 0:
            segment_audio_float = segment_audio.astype(np.float32)
            volume_rms = np.sqrt(np.mean(np.square(segment_audio_float)))
            return 20 * np.log10(volume_rms + 1e-10)
        return -100.0

    def _calculate_pitch(self, segment: AudioSegment) -> float:
        """Tính cao độ giọng nói (Hz) bằng librosa.pyin"""
        start_sample = int(segment.start_time * self.sample_rate)
        end_sample = int(segment.end_time * self.sample_rate)
        segment_audio = self.audio_data[start_sample:end_sample].astype(np.float32)

        if len(segment_audio) == 0:
            return 0.0

        f0, voiced_flag, voiced_prob = librosa.pyin(
            segment_audio,
            fmin=librosa.note_to_hz("C2"),  # ~65 Hz
            fmax=librosa.note_to_hz("C7"),  # ~2093 Hz
            sr=self.sample_rate,
        )

        average_pitch = np.nanmean(f0)
        return float(average_pitch) if not np.isnan(average_pitch) else 0.0

    def _calculate_silence_ratio(self, segment: AudioSegment) -> float:
        """Tính tỷ lệ khoảng lặng trong segment"""
        if segment.duration == 0:
            return 0.0

        segment_speech_duration = 0.0
        for interval in self.non_silent_intervals:
            interval_start_time = interval[0] / self.sample_rate
            interval_end_time = interval[1] / self.sample_rate
            overlap = max(
                0,
                min(segment.end_time, interval_end_time)
                - max(segment.start_time, interval_start_time),
            )
            segment_speech_duration += overlap

        segment_speech_duration = min(segment.duration, segment_speech_duration)
        silence_duration = segment.duration - segment_speech_duration
        return silence_duration / segment.duration

    def _calculate_disfluency_metrics(self, segment: AudioSegment) -> Dict:
        """Đo các chỉ số ngập ngừng tiếng Việt

        Disfluency = từ đệm thực sự (ờ, ừm, à...) + lặp từ
        KHÔNG tính từ lịch sự (dạ, vâng, ạ) là disfluency
        """
        text = segment.text.lower()
        if not text:
            return {"filler_count": 0, "restart_count": 0, "disfluency_rate": 0.0}

        # Tokenize tiếng Việt
        try:
            words = word_tokenize(text, format="text").split()
        except:
            words = re.sub(r"[^\w\s]", " ", text).split()

        total_words = len(words) if words else 1

        # 1. Đếm từ đệm THỰC SỰ (ờ, ừm, à, kiểu...)
        # KHÔNG tính dạ, vâng, ạ là filler
        filler_count = sum(1 for w in words if w in self.FILLER_WORDS)

        # 2. Phát hiện lặp từ (restart): "em em", "dạ dạ", "ờ ờ"
        # Bỏ điều kiện len > 1 để phát hiện "ờ ờ", "ạ ạ"
        restart_count = 0
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                restart_count += 1

        # 3. Tỷ lệ disfluency = (filler + restart) / tổng từ
        disfluency_rate = (filler_count + restart_count) / total_words

        return {
            "filler_count": int(filler_count),
            "restart_count": int(restart_count),
            "disfluency_rate": round(float(disfluency_rate), 3),
        }


class MetadataCalculator:
    """Tính toán metadata của cuộc hội thoại"""

    def __init__(self, dialogue_segments: List[Dict], sales_speaker_id: str):
        self.dialogue_segments = dialogue_segments
        self.sales_speaker_id = sales_speaker_id

    def calculate(self) -> Dict:
        """Tính toán tất cả metadata"""
        total_duration = self._calculate_total_duration()
        turns = self._calculate_turns()
        ratio_sales = self._calculate_sales_ratio(total_duration)

        return {
            "duration": float(total_duration),
            "turns": turns,
            "ratio_sales": float(ratio_sales),
        }

    def _calculate_total_duration(self) -> float:
        """Tính tổng thời lượng cuộc gọi"""
        if not self.dialogue_segments:
            return 0.0
        last_seg = self.dialogue_segments[-1]
        return self._seg_end(last_seg)

    def _calculate_turns(self) -> int:
        """Đếm số lần chuyển người nói"""
        turns = 0
        if len(self.dialogue_segments) > 1:
            for i in range(1, len(self.dialogue_segments)):
                if self.dialogue_segments[i].get("speaker") != self.dialogue_segments[
                    i - 1
                ].get("speaker"):
                    turns += 1
        return turns

    def _calculate_sales_ratio(self, total_duration: float) -> float:
        """Tính tỷ lệ thời gian nói của Sales"""
        sales_duration = sum(
            max(0.0, self._seg_end(seg) - self._seg_start(seg))
            for seg in self.dialogue_segments
            if str(seg.get("speaker")) == str(self.sales_speaker_id)
        )
        return (sales_duration / total_duration) if total_duration > 0 else 0.0

    @staticmethod
    def _seg_start(seg: Dict) -> float:
        return float(seg.get("start", 0.0))

    @staticmethod
    def _seg_end(seg: Dict) -> float:
        return float(seg.get("end", MetadataCalculator._seg_start(seg)))


class SalesPerformanceAnalyzer:
    """Phân tích tổng hợp hiệu suất Sales"""

    @staticmethod
    def analyze_sales_segments(segments: List[Dict]) -> Dict:
        """Tính các chỉ số tổng hợp từ các segment Sales"""
        sales_segments = [s for s in segments if s["speaker"] == "Sales"]

        if not sales_segments:
            return {}

        # 1. Tổng hợp disfluency
        disfluency_rates = [s.get("disfluency_rate", 0) for s in sales_segments]
        total_disfluency_rate = np.mean(disfluency_rates) if disfluency_rates else 0
        # Ngưỡng 0.20 (20%) - phát hiện các đoạn ngập ngừng cao
        high_disfluency_segments = [
            {
                "start": s["start_time"],
                "text": s["text"][:80],
                "rate": s.get("disfluency_rate", 0),
            }
            for s in sales_segments
            if s.get("disfluency_rate", 0) > 0.20
        ]

        # 2. Biến động tốc độ
        spms = [s["speed_spm"] for s in sales_segments if s["speed_spm"] > 0]
        spm_mean = np.mean(spms) if spms else 0
        spm_std = np.std(spms) if len(spms) > 1 else 0
        fast_segments = [
            {"start": s["start_time"], "text": s["text"][:80], "spm": s["speed_spm"]}
            for s in sales_segments
            if s["speed_spm"] > 220
        ]

        pitches = [s["pitch_hz"] for s in sales_segments if s["pitch_hz"] > 0]
        pitch_std = np.std(pitches) if len(pitches) > 1 else 0

        hesitant_responses = SalesPerformanceAnalyzer.analyze_question_responses(
            segments
        )

        return {
            "sales_disfluency": {
                "avg_rate": round(float(total_disfluency_rate), 3),
                "high_segments": high_disfluency_segments[:3],
            },
            "sales_speed": {
                "avg_spm": round(float(spm_mean), 1),
                "spm_std": round(float(spm_std), 1),
                "fast_segments": fast_segments[:3],
            },
            "sales_pitch": {"pitch_std": round(float(pitch_std), 1)},
            "hesitant_responses": hesitant_responses,
        }

    @staticmethod
    def analyze_question_responses(segments: List[Dict]) -> List[Dict]:
        """Phát hiện các turn KH hỏi → Sales trả lời ngập ngừng"""
        question_keywords = [
            "sao",
            "như thế nào",
            "thế nào",
            "có được không",
            "bao nhiêu",
            "giá",
            "tại sao",
            "vì sao",
            "có phải",
        ]

        hesitant_responses = []

        for i in range(len(segments) - 1):
            current = segments[i]
            next_seg = segments[i + 1]

            # Kiểm tra: Customer hỏi → Sales trả lời
            if (
                current["speaker"] == "Customer"
                and next_seg["speaker"] == "Sales"
                and any(kw in current["text"].lower() for kw in question_keywords)
            ):

                # Đo độ trễ và chất lượng response
                response_delay = next_seg["start_time"] - current["end_time"]
                response_disfluency = next_seg.get("disfluency_rate", 0)

                # Text bắt đầu bằng nhiều filler?
                response_text = next_seg["text"].lower()
                starts_with_filler = any(
                    response_text.startswith(f"{filler} ")
                    for filler in ["ờ", "ừm", "à", "em ờ", "dạ ờ"]
                )

                # Đánh giá confidence
                is_hesitant = (
                    (response_delay > 1.5)
                    or (response_disfluency > 0.15)
                    or (starts_with_filler and response_disfluency > 0.10)
                )

                if is_hesitant:
                    hesitant_responses.append(
                        {
                            "question_at": round(current["start_time"], 1),
                            "question": current["text"][:100],
                            "response_delay": round(response_delay, 2),
                            "response_disfluency": round(response_disfluency, 3),
                            "response_preview": next_seg["text"][:100],
                            "starts_with_filler": starts_with_filler,
                        }
                    )

        return hesitant_responses


class AudioFeatureExtractor:
    """Class chính để trích xuất features từ audio"""

    def __init__(self, audio_bytes: bytes):
        self.audio_bytes = audio_bytes
        self.task_id = create_task_id(audio_bytes)

    def _identify_sales_speaker(self, dialogue_segments: List[Dict]) -> str:
        """Xác định Sales dựa trên tổng thời lượng nói"""
        speaker_durations = {}

        for seg in dialogue_segments:
            speaker_id = str(seg.get("speaker", "unknown"))
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            duration = max(0.0, end - start)

            if speaker_id not in speaker_durations:
                speaker_durations[speaker_id] = 0.0
            speaker_durations[speaker_id] += duration

        if speaker_durations:
            sales_speaker_id = max(speaker_durations, key=speaker_durations.get)
            return sales_speaker_id

        return str(dialogue_segments[0].get("speaker", "unknown"))

    async def extract(self) -> Dict:
        """Trích xuất các đặc điểm acoustic và metadata"""
        dialogue_result = await call_dialogue_api(self.audio_bytes, self.task_id)

        if dialogue_result["status"] != 1:
            return {"status": -1, "message": dialogue_result["message"]}

        dialogue_segments = dialogue_result.get("dialogue", [])
        if not dialogue_segments:
            return {
                "status": -1,
                "message": "API không trả về phân đoạn hội thoại nào.",
            }

        # BƯỚC 1: Tự động xác định Sales dựa trên thời lượng nói (preliminary)
        sales_speaker_id = self._identify_sales_speaker(dialogue_segments)
        self.sales_speaker_id = sales_speaker_id

        # BƯỚC 2: Gán nhãn Sales/Customer cho từng segment
        for seg in dialogue_segments:
            seg["speaker"] = (
                "Sales"
                if str(seg.get("speaker")) == str(sales_speaker_id)
                else "Customer"
            )

        # BƯỚC 3: VALIDATE VÀ FIX speaker labels dựa trên NỘI DUNG TEXT
        dialogue_segments, validation_msg = validate_and_fix_speakers(dialogue_segments)
        self.validation_message = validation_msg

        # Log validation result
        print(validation_msg)

        # Load audio data
        audio_data, sample_rate = librosa.load(
            BytesIO(self.audio_bytes), sr=None, dtype=np.float32
        )

        non_silent_intervals = librosa.effects.split(
            audio_data, top_db=25  # Giảm xuống 25 để phát hiện âm thanh yếu hơn
        )

        analyzer = AcousticAnalyzer(audio_data, sample_rate, non_silent_intervals)
        segment_analysis = self._analyze_segments(
            dialogue_segments, sales_speaker_id, analyzer
        )

        metadata_calculator = MetadataCalculator(dialogue_segments, sales_speaker_id)
        metadata = metadata_calculator.calculate()

        # Thêm phân tích tổng hợp Sales
        sales_performance = SalesPerformanceAnalyzer.analyze_sales_segments(
            segment_analysis
        )

        return {
            "status": 1,
            "task_id": self.task_id,
            "segments": segment_analysis,
            "metadata": metadata,
            "sales_performance": sales_performance,
            "validation_info": {
                "message": self.validation_message,
                "speaker_labels_corrected": "⚠️" in self.validation_message,
            },
            "message": "Features and metadata extracted successfully",
        }

    def _analyze_segments(
        self,
        dialogue_segments: List[Dict],
        sales_speaker_id: str,
        analyzer: AcousticAnalyzer,
    ) -> List[Dict]:
        """Phân tích tất cả segments"""
        segment_analysis = []

        for i, seg_data in enumerate(dialogue_segments, start=1):
            # Segment đã có label 'Sales'/'Customer' sau khi validate
            # Không cần dùng AudioSegment để re-label
            speaker_label = seg_data.get("speaker", "unknown")

            start_time = float(seg_data.get("start", 0.0))
            end_time = float(seg_data.get("end", start_time))
            text = seg_data.get("text", "")
            duration = end_time - start_time
            word_count = len(text.split()) if text else 0

            # Tạo AudioSegment để analyze acoustics
            # Nhưng dùng speaker_label đã được validate
            start_sample = int(start_time * analyzer.sample_rate)
            end_sample = int(end_time * analyzer.sample_rate)
            segment_audio = analyzer.audio_data[start_sample:end_sample]

            # Analyze acoustic features
            if duration >= 0.25 and word_count <= 3:
                # Segment có vấn đề
                acoustic_features = {
                    "speed_spm": 0.0,
                    "volume_db": 0.0,
                    "pitch_hz": 0.0,
                    "silence_ratio": 0.0,
                    "filler_count": 0,
                    "restart_count": 0,
                    "disfluency_rate": 0.0,
                }
            else:
                # Tạo temporary AudioSegment object
                temp_seg = type(
                    "obj",
                    (object,),
                    {
                        "start_time": start_time,
                        "end_time": end_time,
                        "speaker_label": speaker_label,
                        "text": text,
                        "duration": duration,
                        "word_count": word_count,
                        "is_corrupted": lambda: False,
                    },
                )
                acoustic_features = analyzer.analyze_segment(temp_seg)

            segment_analysis.append(
                {
                    "segment": i,
                    "speaker": speaker_label,  # Dùng label đã validate
                    "start_time": start_time,
                    "end_time": end_time,
                    "text": text,
                    **acoustic_features,
                }
            )

        return segment_analysis


async def extract_features(audio_bytes: bytes) -> Dict:
    """
    Trích xuất các đặc điểm acoustic và metadata, tự động xác định nhân viên Sales.
    """
    print("=== EXTRACT_FEATURES ĐƯỢC GỌI  ===")
    extractor = AudioFeatureExtractor(audio_bytes)
    result = await extractor.extract()
    print(
        f"=== Sales Speaker ID: {extractor.sales_speaker_id if hasattr(extractor, 'sales_speaker_id') else 'N/A'} ==="
    )
    return result
