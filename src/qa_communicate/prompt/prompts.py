import json

# Template prompt được lưu trữ dưới dạng một hằng số private
_QA_EVALUATION_TEMPLATE = """
# NHIỆM VỤ
Phân tích cuộc gọi sales dựa trên transcript và acoustic features, sau đó đánh giá kỹ năng giao tiếp của Sales theo các tiêu chí đánh giá 
và tính điểm tổng theo công thức quy định.

# TIÊU CHÍ ĐÁNH GIÁ 

## TIÊU CHÍ 1 : CHÀO/XƯNG DANH
### TIÊU CHUẨN ĐẠT
-Có xưng danh trong 1,2 segment đầu tiên
### TIÊU CHUẨN KHÔNG ĐẠT 
-Không xưng danh
-Xưng danh nhưng khách hàng không nghe được và hỏi lại

## TIÊU CHÍ 2 : KỸ NĂNG NÓI
### TIÊU CHUẨN ĐẠT 
-Giọng nói rõ ràng, âm lượng và cường độ vừa phải (Tốc độ nói của SALES < 220 SPM nếu có vài đoạn có SPM lớn hơn thì hãy DỰA VÀO NGỮ CẢNH CHỨ ĐỪNG VỘI ĐÁNH GIÁ)
-Giọng nói có điểm nhấn (Dựa vào pitch_hz,silence_ratio và nên đưa ra dẫn chứng trong phần giải thích)
-Có sự nhấn nhá trong giao tiếp
### TIÊU CHUẨN KHÔNG ĐẠT
-Nói quá nhanh khiến khách hàng NGHE KHÔNG RÕ VÀ YÊU CẦU NHẮC LẠI
-Cách diễn đạt không trôi chảy, ngập ngừng nhiều,NÓI KHÔNG LƯU LOÁT
-GIỌNG NÓI ĐỀU ĐỀU,KHÔNG TẠO ĐIỂM NHẤN (Dựa vào pitch_hz,silence_ratio)
-Nói quá nhỏ khiến khác không nghe rõ và yêu cầu nhắc lại (Ví dụ : chị nói nhỏ quá không nghe rõ gì))
-Lỗi về sự trôi chảy (Phân biệt rõ 2 loại):
    1. **Ngập ngừng/Tìm từ:** Diễn đạt không trôi chảy, lặp lại từ, ngập ngừng nhiều với các từ đệm (ví dụ: "ờm", "à", "uh") hoặc các khoảng lặng dài bất thường để suy nghĩ.
    2. **Hụt hơi/Câu dài:** Giao tiếp bị ngắt quãng, gãy vụn. Dấu hiệu là sales nói một câu quá dài, sau đó bị hụt hơi và phải **ngắt nghỉ đột ngột ở những vị trí không phù hợp** (ví dụ: đang nói giữa một cụm danh từ, động từ) để lấy hơi. Điều này khiến câu nói bị cắt ngang một cách thiếu chuyên nghiệp.
### CHÚ Ý 
-TẬP TRUNG VÀO CÁC SEGMENT CỦA SALES
-KHÔNG ĐƯỢC DỰA VÀO SEGMENT CỦA CUSTOMER ĐỀ ĐÁNH GIÁ KỸ NĂNG NÓI CỦA SALES 
    
## TIÊU CHÍ 3 : KĨ NĂNG NGHE,TRẤN AN
### TIÊU CHUẨN ĐẠT
-Tập trung thể hiện sự đồng cảm,lắng nghe những thông tin khách hàng chia sẻ ( tập trung vào text)
-Vấn an, cảm thông với những khách hàng đang chưa hài lòng
### TIÊU CHUẨN KHÔNG ĐẠT
-Khách hàng chia sẻ nhưng sale không thể hiện sự đồng cảm cùng,thông tin chia sẻ bị bỏ quên,không hào hứng tham gia
-Các vấn đề lỗi và tạo sự trải nghiệm chưa hài lòng từ khách hàng,sale chưa thể hiện sự trấn an

## TIÊU CHÍ 4 : THÁI ĐỘ GIAO TIẾP 
### TIÊU CHUẨN ĐẠT
-Ngôn ngữ giao tiếp chuẩn mực,thể hiện sự tôn trọng với khách hàng
-Giải quyết triệt để vấn đề của khách hàng đứng trên quan điểm người sử dụng dịch vụ
### TIÊU CHUẨN KHÔNG ĐẠT
- Có thái độ,ngữ điệu trong cuộc gọi không vui vẻ nhiệt tình
-Ngôn từ thiếu chuẩn mực,chưa thể hiện sự tôn trọng với khách hàng 

# ĐỊNH NGHĨA MỨC LỖI
M1: Thực hiện đủ thông tin (chào, xưng danh) nhưng chưa trôi chảy (nhiều disfluency_count) hoặc thiếu nhiệt tình (giọng đều đều, pitch_stddev thấp).
      Nói nhỏ (volume_db thấp) hoặc nói nhanh (speed_spm cao).
      Câu từ thiếu chủ ngữ, vị ngữ, ngôn từ bình dân (phân tích text)
M2: Cao giọng, mỉa mai (phân tích text + pitch_stddev cao), thể hiện sự hiểu biết hơn KH, thiếu trách nhiệm.
      Không thực hiện quy định tiếp nhận cuộc gọi (không chào hỏi/ chào kết thúc khi KH đã chào sales) (phân tích text segments đầu/cuối).
      Cung cấp thông tin không quan tâm đến KH có hiểu hay không (ví dụ: ratio_sales quá cao, nói liên tục).
M3: Khai thác lại thông tin lần 2, kết thúc cuộc gọi vẫn không phát hiện vấn đề của KH (phân tích text).
      Có cử chỉ, thái độ, ngôn ngữ hoặc hành vi không lịch sự, thiếu văn hoá, thiếu tôn trọng KH (ví dụ: politeness_count == 0 và text cộc lốc).

# DỮ LIỆU CUỘC GỌI
```json
{call_data_str}

# YÊU CẦU
Hãy nghĩ từng bước trước khi đưa ra đánh giá từng tiêu chí
Đánh giá các kĩ năng của Sales 
# OUTPUT FORMAT
```json
{{
  "chao_xung_danh": <int: 0/1>,
  "ky_nang_noi": <int: 0/1>,
  "ky_nang_nghe": <int: 0/1>,
  "thai_do": <int: 0/1>,
  "muc_loi": <string: "Không"|"M1"|"M2"|"M3">,
  "ly_do": <string: giải thích kĩ thành các gạch đầu dòng , trích dẫn rõ và đúng nội dung segment ứng với các tiêu chí tương ứng (ví dụ : tốc độ nói của sales nhanh không được trích xuất segment của customer) >
}}
```
"""


def build_qa_prompt(call_data: dict) -> str:
    """Xây dựng prompt chấm điểm QA bằng cách chèn dữ liệu cuộc gọi vào template."""
    call_data_str = json.dumps(call_data, indent=2, ensure_ascii=False)
    return _QA_EVALUATION_TEMPLATE.format(call_data_str=call_data_str)
