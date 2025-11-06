# -*- coding: utf-8 -*-
"""
Speaker Validator - Tự động phát hiện và sửa lỗi phân loại Sales/Customer
Không dựa vào LLM, chỉ dùng rule-based trên nội dung text
"""

import re
from typing import List, Dict, Tuple
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class SpeakerValidator:
    """
    Tự động validate và fix speaker labels
    
    Nguyên tắc:
    1. Phân tích 3-5 segment đầu tiên (quan trọng nhất)
    2. Tính "Sales score" vs "Customer score" cho mỗi segment
    3. Nếu phát hiện đảo ngược → swap ALL segments
    """
    
    # Dấu hiệu CHẮC CHẮN là Sales
    SALES_STRONG_INDICATORS = [
        # Xưng danh
        r'em l\u00e0 \w+',  # "em là Hương"
        r'em t\u00ean l\u00e0',
        r'b\u00ean em',  # "bên em"
        r't\u1eeb \w+ (cloud|bizfly|vccorp|viettel)',  # "từ Bizfly"
        r'c\u00f4ng ty \w+',
        
        # Giới thiệu sản phẩm
        r'em s\u1ebd h\u1ed7 tr\u1ee3',
        r'em g\u1eedi .*(b\u00e1o gi\u00e1|th\u00f4ng tin)',
        r'em xin ph\u00e9p',
        r'cho em h\u1ecfi',
        
        # Chốt đơn
        r'anh quan t\u00e2m.*kh\u00f4ng',
        r'anh c\u00f3 nhu c\u1ea7u',
        r'b\u00ean em c\u00f3 (g\u00f3i|d\u1ecbch v\u1ee5)',
    ]
    
    # Dấu hiệu YẾU là Sales (tính điểm thấp hơn)
    SALES_WEAK_INDICATORS = [
        r'\bem\b',  # Xưng "em"
        r'\banh \u1ea1\b',  # "anh ạ"
        r'\bch\u1ecb \u1ea1\b',  # "chị ạ"
        r'd\u1ea1 v\u00e2ng',
    ]
    
    # Dấu hiệu CHẮC CHẮN là Customer
    CUSTOMER_STRONG_INDICATORS = [
        # Hỏi giá, tính năng
        r'bao nhi\u00eau ti\u1ec1n',
        r'gi\u00e1.*th\u1ebf n\u00e0o',
        r'c\u00f3 .*kh\u00f4ng',  # "có tính năng X không"
        r'.*\u0111\u01b0\u1ee3c kh\u00f4ng',  # "dùng thử được không"
        r'c\u00f3 \u0111\u1eaft kh\u00f4ng',
        
        # Phàn nàn, thắc mắc
        r't\u1ea1i sao l\u1ea1i',
        r'sao l\u1ea1i',
        r'th\u1ebf th\u00ec',
        r'nh\u01b0ng m\u00e0',
        
        # Response thụ động
        r'^\u01b0$',  # Chỉ "ừ"
        r'^\u0111\u01b0\u1ee3c$',  # Chỉ "được"
        r'^ok$',
        r'^oke$',
    ]
    
    # Dấu hiệu YẾU là Customer
    CUSTOMER_WEAK_INDICATORS = [
        r'\banh\b',  # Xưng "anh" (không có "ạ" phía sau)
        r'\bm\u00ecnh\b',  # "mình"
        r'\bt\u00f4i\b',  # "tôi"
    ]
    
    def __init__(self):
        # Compile regex patterns
        self.sales_strong_patterns = [re.compile(p, re.IGNORECASE) for p in self.SALES_STRONG_INDICATORS]
        self.sales_weak_patterns = [re.compile(p, re.IGNORECASE) for p in self.SALES_WEAK_INDICATORS]
        self.customer_strong_patterns = [re.compile(p, re.IGNORECASE) for p in self.CUSTOMER_STRONG_INDICATORS]
        self.customer_weak_patterns = [re.compile(p, re.IGNORECASE) for p in self.CUSTOMER_WEAK_INDICATORS]
    
    def calculate_speaker_score(self, text: str) -> Tuple[float, float]:
        """
        Tính điểm Sales vs Customer cho một segment
        
        Returns:
            (sales_score, customer_score)
        """
        if not text or len(text.strip()) < 3:
            return (0.0, 0.0)
        
        text = text.lower()
        
        sales_score = 0.0
        customer_score = 0.0
        
        # Strong indicators (5 điểm)
        for pattern in self.sales_strong_patterns:
            if pattern.search(text):
                sales_score += 5.0
        
        for pattern in self.customer_strong_patterns:
            if pattern.search(text):
                customer_score += 5.0
        
        # Weak indicators (1 điểm)
        for pattern in self.sales_weak_patterns:
            if pattern.search(text):
                sales_score += 1.0
        
        for pattern in self.customer_weak_patterns:
            if pattern.search(text):
                customer_score += 1.0
        
        return (sales_score, customer_score)
    
    def validate_segments(self, segments: List[Dict]) -> Dict:
        """
        Validate speaker labels cho toàn bộ segments
        
        Returns:
            {
                'is_swapped': bool,
                'confidence': float,
                'evidence': List[str],
                'corrected_segments': List[Dict]
            }
        """
        if not segments or len(segments) < 3:
            return {
                'is_swapped': False,
                'confidence': 0.0,
                'evidence': ['Không đủ segment để validate'],
                'corrected_segments': segments
            }
        
        # Phân tích 5 segment đầu tiên (quan trọng nhất)
        check_segments = segments[:min(5, len(segments))]
        
        evidence = []
        total_mismatch_score = 0.0
        
        for i, seg in enumerate(check_segments, 1):
            current_label = seg.get('speaker', 'unknown')
            text = seg.get('text', '')
            
            sales_score, customer_score = self.calculate_speaker_score(text)
            
            # Kiểm tra xem có mâu thuẫn không
            if current_label == 'Sales' and customer_score > sales_score + 3:
                # Được gán là Sales nhưng nội dung giống Customer
                mismatch = customer_score - sales_score
                total_mismatch_score += mismatch
                evidence.append(
                    f"Segment {i} (gán 'Sales'): \"{text[:50]}...\" "
                    f"→ Customer score={customer_score:.0f}, Sales score={sales_score:.0f} "
                    f"→ Có vẻ là CUSTOMER!"
                )
            
            elif current_label == 'Customer' and sales_score > customer_score + 3:
                # Được gán là Customer nhưng nội dung giống Sales
                mismatch = sales_score - customer_score
                total_mismatch_score += mismatch
                evidence.append(
                    f"Segment {i} (gán 'Customer'): \"{text[:50]}...\" "
                    f"→ Sales score={sales_score:.0f}, Customer score={customer_score:.0f} "
                    f"→ Có vẻ là SALES!"
                )
        
        # Quyết định có swap không
        # Nếu có >= 2 segment mâu thuẫn với mismatch_score > 10 → swap
        is_swapped = len(evidence) >= 2 and total_mismatch_score > 10
        
        corrected_segments = segments
        if is_swapped:
            logger.warning(f"⚠️ Phát hiện speaker labels bị SWAP! Đang sửa...")
            corrected_segments = self._swap_all_speakers(segments)
        
        return {
            'is_swapped': is_swapped,
            'confidence': min(total_mismatch_score / 20.0, 1.0),  # Normalize to 0-1
            'evidence': evidence,
            'corrected_segments': corrected_segments
        }
    
    def _swap_all_speakers(self, segments: List[Dict]) -> List[Dict]:
        """Đảo ngược tất cả speaker labels"""
        corrected = []
        for seg in segments:
            new_seg = seg.copy()
            if seg['speaker'] == 'Sales':
                new_seg['speaker'] = 'Customer'
            elif seg['speaker'] == 'Customer':
                new_seg['speaker'] = 'Sales'
            corrected.append(new_seg)
        return corrected
    
    def get_validation_summary(self, validation_result: Dict) -> str:
        """Tạo summary cho validation result (để log hoặc gửi cho LLM)"""
        if not validation_result['is_swapped']:
            return "✅ Speaker labels hợp lệ"
        
        summary = "⚠️ ĐÃ PHÁT HIỆN VÀ SỬA LỖI PHÂN LOẠI SPEAKER\n\n"
        summary += "Bằng chứng:\n"
        for ev in validation_result['evidence']:
            summary += f"- {ev}\n"
        summary += f"\n→ Kết luận: API đã gán nhãn NGƯỢC. Đã swap tất cả segments.\n"
        summary += f"→ Độ tin cậy: {validation_result['confidence']:.0%}\n"
        
        return summary


def validate_and_fix_speakers(segments: List[Dict]) -> Tuple[List[Dict], str]:
    """
    Convenience function để validate và fix speakers
    
    Args:
        segments: List of segment dicts với keys: speaker, text, ...
    
    Returns:
        (corrected_segments, validation_message)
    """
    validator = SpeakerValidator()
    result = validator.validate_segments(segments)
    
    validation_msg = validator.get_validation_summary(result)
    
    if result['is_swapped']:
        logger.warning(validation_msg)
    
    return result['corrected_segments'], validation_msg
