import json


_QA_EVALUATION_TEMPLATE = """
# NHIỆM VỤ
Phân tích cuộc gọi sales dựa trên transcript, acoustic features, và các chỉ số ngập ngừng/tự tin, 
sau đó đánh giá kỹ năng giao tiếp của Sales theo tiêu chí **NGHIÊM NGẶT**.

# DỮ LIỆU CUỘC GỌI
```json
{call_data_str}
```

# TIÊU CHÍ ĐÁNH GIÁ (PHẢI TUÂN THỦ CHẶT CHẼ)
## TIÊU CHÍ 1: CHÀO/XƯNG DANH

### Tiêu chuẩn ĐẠT (1 điểm)
- Có xưng danh rõ ràng trong **bất kỳ segment  nào trong 4 segment đầu tiên**
- Khách hàng KHÔNG hỏi lại "ai gọi đấy" / "bên nào vậy" / "ai đó"

### Tiêu chuẩn KHÔNG ĐẠT (0 điểm)
- Không xưng danh hoặc xưng danh quá muộn (sau segment thứ 4)
- Khách hàng phải hỏi lại vì không biết ai gọi

### Lưu ý ĐẶC BIỆT (BẮT BUỘC TUÂN THỦ)

**TRƯỚNG HỢP ĐẶC BIỆT - KHÔNG TRỪ ĐIỂM:**

1. **Sales nói "A lô" trước, rồi xưng danh sau** → ĐẠT
   - VD: Seg 1 (Sales): "A lô ạ" → Seg 2 (KH): "A lô" → Seg 3 (Sales): "Dạ em là Hương từ Bizfly"
   - Lý do: Đây là cách tiếp nhận cuộc gọi chuẩn mực, bắt máy trước rồi xưng danh

2. **KH bắt máy trước, Sales xưng danh ngay** → ĐẠT
   - VD: Seg 1 (KH): "A lô" → Seg 2 (Sales): "Chào anh, em là Hương bên Bizfly"
   - Lý do: Xưng danh ngay sau khi KH bắt máy

**CÁCH KIỂM TRA ĐÚNG:**
- BƯỚC 1: Liệt kê TẤT CẢ segment của Sales trong 4 segment đầu tiên
- BƯỚC 2: Kiểm tra XEM CÓ BẤT KỲ segment Sales nào chứa xưng danh:
  + Tên: "em là [Tên]", "mình là [Tên]", "[Tên] bên [Công ty]"
  + Công ty: "từ Bizfly", "bên Bizfly", "của Vccorp", 
  + Hoặc cả hai: "em là Hương bên Bizfly Cloud"
- BƯỚC 3: Nếu CÓ → ĐẠT, nếu KHÔNG → KHÔNG ĐẠT
- BƯỚC 5 : KIỂM TRA KĨ CẢ LỜI THOẠI CỦA CUSTOMER NẾU THẤY SALES KHÔNG XƯNG DANH ( CÓ THỂ LÀ HỆ THỐNG NHẬN DIỆN NHẦM SALE VÀ CUSTOMER)


**LỖI THƯỚNG GẶP CẦN TRÁNH:**
- ❌ SAI: Chỉ xem segment 1 của Sales → bỏ qua segment 3, 4
- ❌ SAI: Cho rằng "A lô" = không xưng danh → sai, phải xem các segment sau
- ✅ ĐÚNG: Xem TẤT CẢ segment Sales từ 1-4, tìm xưng danh ở BẤT KÌ segment nào

**BẮT BUỘC TRÍCH DẪN:**
- Nếu ĐẠT: Trích segment có xưng danh với timestamp
- Nếu KHÔNG ĐẠT: Liệt kê TẤT CẢ segment Sales từ 1-4 để chứng minh không có xưng danh
- BẮT BUỘC trích dẫn segment + timestamp làm bằng chứng

---

## TIÊU CHÍ 2: KỸ NĂNG NÓI

### 🎯 TRIẾT LÝ ĐÁNH GIÁ

**Bạn là một QA chuyên nghiệp, đánh giá giống cách QA thực sự làm việc:**

1. **Diễn đạt tự nhiên, không dùng số liệu cứng nhắc:**
   - "Hơi nhanh" thay vì "230 từ/phút"
   - "Còn ngập ngừng" thay vì "disfluency 18%"
   - "Giọng đều đều chưa tạo điểm nhấn" thay vì "pitch_std = 11"

2. **Kết hợp "tuy nhiên" khi đánh giá:**
   - "Giao tiếp nhẹ nhàng, TUY NHIÊN hơi nhanh" → vẫn trừ điểm
   - "Câu từ tự tin, TUY NHIÊN ngữ điệu hơi cứng" → M1

3. **Nhìn tổng thể, không tách rời:**
   - Đầu cuộc gọi ngập ngừng, sau lưu loát → ĐẠT
   - Nhanh nhưng rõ ràng, KH không khó khăn → có thể ĐẠT
   - Chậm nhưng ngắt nghỉ không phù hợp → vẫn M1

4. **Ghi chú cải thiện cụ thể:**
   - Không chỉ "sai gì" mà còn "nên làm gì"
   - VD: "Nên chậm lại và tạo điểm nhấn đúng thông tin trọng tâm"

### Dữ liệu đầu vào cần kiểm tra
1. **sales_performance.sales_disfluency.avg_rate**: Tỷ lệ từ đệm THỰC SỰ (ờ, ừm, à, kiểu...) + lặp từ
   - **Lưu ý**: KHÔNG tính 'dạ', 'vâng', 'ạ' là filler (là từ lịch sự tiếng Việt)
2. **sales_performance.sales_disfluency.high_segments**: Các đoạn có tỷ lệ ngập ngừng cao
3. **sales_performance.sales_speed.avg_spm**: Tốc độ nói trung bình (từ/phút)
4. **sales_performance.sales_speed.spm_std**: Độ lệch chuẩn tốc độ (cao = lúc nhanh lúc chậm)
5. **sales_performance.sales_speed.fast_segments**: Các đoản SPM > 220
6. **sales_performance.sales_pitch.pitch_std**: Độ biến động cao độ giọng (thấp < 15 = đơn điệu)
7. **sales_performance.hesitant_responses**: Các lần Sales trả lời KH không tự tin

### Tiêu chuẩn ĐẠT (1 điểm) - ĐÁNH GIÁ TỔNG THỂ

✅ **Không ngập ngừng đáng kể**:
  - `sales_disfluency.avg_rate < 0.15` (< 15% từ là filler thực sự: ờ, ừm, à, kiểu... + lặp từ)
  - HOẶC nếu 0.15-0.20 nhưng chỉ tập trung ở đầu cuộc gọi (phần sau lưu loát hơn) → vẫn ĐẠT
  

✅ **Tốc độ hợp lý, dễ nghe**:
  - `sales_speed.avg_spm < 230` (cho phép nói nhanh vừa phải nếu KH không phàn nàn)
  - `len(sales_speed.fast_segments) <= 3` (cho phép vài đoạn nhanh nếu là đoạn giải thích quen thuộc)

✅ **Giọng có điểm nhấn hoặc nhiệt tình**:
  - `sales_pitch.pitch_std >= 12` (nới từ 15 → 12, chấp nhận giọng bình thường nếu thái độ tốt)
  - HOẶC nếu pitch_std thấp nhưng KH phản hồi tích cực

✅ **Trả lời KH tự tin**:
  - `len(hesitant_responses) <= 1` (cho phép 1 lần ngập ngừng nếu câu hỏi khó/bất ngờ)
  - Nếu có 2 lần nhưng Sales sau đó tự điều chỉnh và trả lời đúng 



### Tiêu chuẩn KHÔNG ĐẠT (0 điểm) - XEM XÉT NGỮ CẢNH TRƯỚC KHI KẾT LUẬN

**⚠️ TRƯỚC KHI TRỪ ĐIỂM, HỎI BẢN THÂN:**
- Điều này có THỰC SỰ làm giảm chất lượng cuộc gọi không?
- Khách hàng có bị ảnh hưởng tiêu cực không? 
- Có yếu tố giảm nhẹ không? (câu hỏi khó, chủ đề phức tạp, đầu cuộc gọi còn bỡ ngỡ)

❌ **Ngập ngừng THỰC SỰ ẢNH HƯỞNG**:

  ** - Ngập ngừng đáng kể :**
  - `0.20 <= sales_disfluency.avg_rate < 0.30` VÀ ngập ngừng xuyên suốt cuộc gọi (không cải thiện) → **M1**
  - Hoặc `len(sales_disfluency.high_segments) >= 3` và đều ở các thời điểm quan trọng (giải thích sản phẩm, trả lời KH)
  
  → **Giải thích cho QA**: Sales nói chưa lưu loát, nhiều từ đệm 'ờ', 'ừm', 'à', 'kiểu' xuyên suốt cuộc gọi, ảnh hưởng tính chuyên nghiệp.
  → **Bằng chứng**: Trích 2-3 đoạn điển hình có 'ờ', 'ừm', lặp từ với thời điểm và nội dung.
  

  
❌ **Nói nhanh GÂY KHÓ NGHE**:

  ** - Nói nhanh vừa phải nhưng có dấu hiệu KH khó theo:**
  - `230 <= sales_speed.avg_spm < 250` VÀ KH có dấu hiệu không theo kịp (hỏi lại nhiều lần, "hả?", "sao?") → **M1**
  - Hoặc `len(sales_speed.fast_segments) >= 4` và tập trung ở phần giải thích quan trọng
  
  → **Giải thích cho QA**: Giao tiếp hơi nhanh, KH có dấu hiệu khó theo dõi, cần điều chỉnh nhịp độ.
  → **Bằng chứng**: Nêu tốc độ trung bình (VD: 235 từ/phút) và trích segment KH hỏi lại.

  ** - Nói rất nhanh :**
  - `sales_speed.avg_spm >= 250` → **M1** (tùy mức độ ảnh hưởng)
  - Hoặc có đoạn > 280 SPM ở phần quan trọng
  
  → **Giải thích cho QA**: Sales nói quá nhanh, khách hàng khó tiếp thu thông tin.

  ** - Tốc độ thất thường nghiêm trọng :**
  - `sales_speed.spm_std >= 60` → **M2** (chênh lệch quá lớn, gây rối)
  
  → **Giải thích cho QA**: Tốc độ biến thiên quá lớn (lúc rất chậm lúc rất nhanh), gây khó chịu cho người nghe.

❌ **Giọng đơn điệu VÀ ẢNH HƯỞNG TIÊU CỰC**:
  - `sales_pitch.pitch_std < 10` VÀ cuộc gọi dài > 3 phút → **M1** (giọng quá phẳng, thiếu nhiệt tình)
  
  → **Giải thích cho QA**: Giọng nói khá phẳng, thiếu điểm nhấn, có thể khiến KH cảm thấy thiếu nhiệt tình.
  → **Bằng chứng**: Nêu giá trị pitch_std (VD: "9 Hz - thấp, giọng ít biến thiên") + mô tả ảnh hưởng.

❌ **Mất tự tin ĐÁNG KỂ khi KH hỏi** (Lỗi M1 ):
  - `len(hesitant_responses) >= 3` VÀ các câu hỏi đều là câu hỏi cơ bản →
  - `len(hesitant_responses) == 2` VÀ ở câu hỏi quan trọng (giá, tính năng chính) → **Lỗi M1**
  - **KHÔNG trừ điểm** nếu chỉ 1 lần ngập ngừng ở câu hỏi khó/bất ngờ
  
  →: Sales trả lời chậm hoặc ngập ngừng nhiều khi KH hỏi, thể hiện chưa nắm vững thông tin.
  → **Bằng chứng**: Trích dẫn từ `hesitant_responses`: câu hỏi của KH (giây X), câu trả lời ngập ngừng của Sales.

❌ **KH phàn nàn về cách nói** :
  - Có segment Customer chứa: "nói nhanh quá", "nhỏ quá", "không nghe rõ", "chậm lại", "to lên" → **TỰ ĐỘNG M1-M2**
  
  → **Bằng chứng**: Trích segment Customer cụ thể (giây + nội dung).

❌ **Quá cứng nhắc trong giao tiếp** :
  - Các segments của sale có bố cục giống nhau nhiều ( Dạ vầng ạ ....... ở nhiều câu liên tiếp)
  ->  **Giải thích cho QA** "Cách giao tiếp quá cứng nhắc cần mềm mỏng hơn "

### QUY TẮC ƯU TIÊN (BẮT BUỘC TUÂN THỦ)

1. **ĐÁNH GIÁ TỔNG THỂ - KHÔNG CỨNG NHẮC**: 
   - Xem xét toàn bộ cuộc gọi, không chỉ dựa vào 1-2 con số
  

2. **BẮT BUỘC trích dẫn bằng chứng cụ thể**:
   - Với disfluency: 2-3 đoạn điển hình (giây + nội dung)
   - Với tốc độ: Liệt kê đoạn nhanh VÀ mô tả ảnh hưởng đến KH
   - Với hesitant: Trích câu hỏi KH + câu trả lời Sales + đánh giá mức độ nghiêm trọng

### ⚠️ XỬ LÝ KHI NGHI NGỜ PHÂN LOẠI SAI SPEAKER

**VẤN ĐỀ NGHIÊM TRỌNG**: API dialogue đôi khi gộp lời thoại Sales và Customer vào 1 segment, hoặc phân loại sai hoàn toàn.

**🔴 QUY TẮC QUAN TRỌNG NHẤT:**
**Nếu phát hiện 1 segment phân loại sai → TẤT CẢ segment đều bị phân loại NGƯỢC LẠI (Sales ↔ Customer)**

Ví dụ:
- Nếu segment "Sales" có nội dung như KH → thực tế TẤT CẢ segment "Sales" là của Customer, và ngược lại
- Đây là lỗi hệ thống của API, không phải lỗi nhỏ lẻ

### DẤU HIỆU PHÁT HIỆN PHÂN LOẠI SAI

**Dấu hiệu segment "Sales" THỰC RA là Customer:**
1. Hỏi về giá, tính năng: "bao nhiêu tiền?", "có tính năng X không?", "dùng thử được không?"
2. Không xưng "em", không gọi "anh/chị" (xưng "anh", "mình", "tôi")
3. Không đề cập công ty/sản phẩm bán (Bizfly, VCCorp, Cloud Server...)
4. Ngôn ngữ thụ động: "ừ", "được", "để xem", "chưa rõ", "anh nghĩ sao?"
5. Thắc mắc, phàn nàn: "sao lại...", "tại sao...", "có đắt không?"

**Dấu hiệu segment "Customer" THỰC RA là Sales:**
1. Giải thích sản phẩm: "em sẽ hỗ trợ anh", "gói này có tính năng...", "bên em có ưu đãi..."
2. Xưng "em", gọi "anh/chị"
3. Đề cập công ty: "bên em", "Bizfly", "VCCorp", "team em"
4. Chốt đơn, hỏi nhu cầu: "anh quan tâm gói nào?", "em gửi báo giá cho anh nhé"
5. Xưng danh đầy đủ họ tên , công ty trong các segment đầu .

### HÀNH ĐỘNG BẮT BUỘC KHI PHÁT HIỆN

**BƯỚC 1: KIỂM TRA KỸ 3-4 segment đầu tiên**
- Đọc kỹ nội dung text của segment được gán nhãn "Sales"
- Xác định: Có phải nội dung như KH không?

**BƯỚC 2: NẾU PHÁT HIỆN DẤU HIỆU SAI → ĐẢO NGƯỢC TOÀN BỘ**
- **QUAN TRỌNG**: Không chỉ loại bỏ segment đó
- **PHẢI ĐẢO NGƯỢC HOÀN TOÀN**: Tất cả segment "Sales" → Customer, tất cả segment "Customer" → Sales
- Lý do: Đây là lỗi hệ thống API gán nhãn ngược hoàn toàn, không phải lỗi từng segment

**BƯỚC 3: ĐÁNH GIÁ LẠI SAU KHI ĐẢO**
- Đánh giá disfluency, speed, pitch dựa trên segment THỰC SỰ của Sales (sau khi đảo)
- Ghi chú: "⚠️ Đã phát hiện API phân loại ngược speaker. Đã đảo toàn bộ: segments 'Sales' → Customer, 'Customer' → Sales"

**BƯỚC 4: TRÍCH DẪN BẰNG CHỨNG**
```
⚠️ CẢNH BÁO: Phát hiện API phân loại ngược speaker

Bằng chứng:
- Segment 2 (gán nhãn "Sales"): "bao nhiêu tiền một tháng?" → Đây là câu hỏi của KHÁCH HÀNG
- Segment 5 (gán nhãn "Customer"): "dạ bên em có gói 500k/tháng ạ" → Đây là lời SALES

→ Kết luận: API đã gán nhãn ngược. Đã ĐẢO TOÀN BỘ để đánh giá đúng.

Sau khi đảo:
- Disfluency_rate của Sales thực: 0.12 (chấp nhận được)
- Speed của Sales thực: 195 từ/phút (tốt)
```

### Lưu ý quan trọng
- **'Dạ', 'vâng', 'ạ' KHÔNG phải là filler**: Là từ lịch sự tiếng Việt chuẩn mực, không trừ điểm
- **Từ filler thực sự**: 'ờ', 'ừm', 'à', 'kiểu', 'dạng', 'ấy là'...
- **Đọc kỹ text trước khi tin vào speaker_label**: API có thể sai, hãy dùng common sense
- **Ưu tiên logic hơn dữ liệu**: Nếu dữ liệu không hợp lý, phải kiểm tra và điều chỉnh

---
## TIÊu CHÍ 3: KỲ NĂNG NGHE, TRẤN AN ĐỒNG CẢM

### ⚠️ QUY TẮC ÁP DỤNG (BẮT BUỘC ĐỌC TRƯỚC)

**CHỈ ĐÁNH GIÁ TIÊU CHÍ NÀY KHI:**
1. KH có khiếu nại / phàn nàn / than phiền về dịch vụ/sản phẩm
2. KH chia sẻ vấn đề cá nhân (bệnh tật, khó khăn...)
3. KH thể hiện cảm xúc tiêu cực (bực mình, thất vọng...)

**KHÔNG ĐÁNH GIÁ TIÊU CHÍ NÀY KHI:**
- CG bán hàng thuần túy (KH không phàn nàn, chỉ hỏi thông tin)
- CG tư vấn kỹ thuật (KH chỉ hỏi cách sử dụng)
- CG ngắn, không có tương tác sâu

→ **Nếu KHÔNG có ngữ cảnh khiếu nại/than phiền**: Ghi "Không áp dụng" trong phần nhận xét và CHO ĐIỂM 1 

---

### Tiêu chuẩn ĐẠT (1 điểm) - CHỈ XÉT KHI CÓ NGỮ CẢNH PHIỀN/KHIẾU NẠI

✅ **Lắng nghe, ghi nhận thông tin**:
  - Sales lặp lại/xác nhận thông tin KH chia sẻ
  - Không bỏ sót thông tin quan trọng KH đã nêu

✅ **Thể hiện sự đồng cảm, trấn an**:
  - Dùng ngôn ngữ: "em hiểu ạ", "anh đừng lo", "em sẽ hỗ trợ ngay"
  - Giọng nói nhẹ nhàng, chậm rãi khi KH bực mình

### Tiêu chuẩn KHÔNG ĐẠT (0 điểm) - CHỈ TRỪ ĐIỂM KHI CÓ NGỮ CẢNH PHIỀN/KHIỄU NẠI

❌ **Bỏ sót thông tin**:
  - KH chia sẻ nhưng Sales không ghi nhận, bỏ quên
  - Cứ hỏi lại thông tin KH đã cung cấp

❌ **Không trấn an khi KH phàn nàn**:
  - KH bực mình/than phiền nhưng Sales không có lời trấn an
  - Phản ứng khô khan, chỉ nói "vâng dạ" rồi chuyển sang vấn đề khác
- KH phàn nàn nhưng Sales không trấn an

---

## TIÊU CHÍ 4: THÁI ĐỘ GIAO TIẾP
### Tiêu chuẩn ĐẠT
- Ngôn ngữ chuẩn mực, thể hiện tôn trọng khách hàng
- Giải quyết vấn đề đứng góc độ KH

### Tiêu chuẩn KHÔNG ĐẠT
- Thái độ không nhiệt tình
- Ngôn từ cộc lốc, thiếu tôn trọng

---

# ĐỊNH NGHĨA MỨC LỖI (BẮT BUỘC GÁN ĐÚNG)

**MỨC 1 - Lỗi nhỏ** :
- **Lỗi kỹ năng giao tiếp**:
  + Ngập ngừng vừa phải: `0.15 <= sales_disfluency.avg_rate < 0.25`
  + Thiếu nhiệt tình, giọng đơn điệu: `sales_pitch.pitch_std < 15`
  + Nói nhanh hoặc tốc độ không đều: `sales_speed.avg_spm >= 240` HOAC `sales_speed.spm_std >= 50` HOAC `len(fast_segments) >= 3`
  + Có 1 lần trả lời KH ngập ngừng: `len(hesitant_responses) == 1`
  + Giọng địa phương (nếu phát hiện từ text)
  + Câu từ thiếu chủ ngữ, vị ngữ, ngôn từ giao tiếp bình dân (phân tích text)

**MỨC 2 - Lỗi vừa**:
- **Lỗi kỹ năng giao tiếp**:
  + Cao giọng, mỉa mai, thể hiện hiểu biết hơn KH, thiếu trách nhiệm
  + Cung cấp thông tin không quan tâm KH có hiểu hay không 
  + Không hiểu, không biết xác nhận dẫn đến hiểu sai và cung cấp thông tin sai
  + Bỏ qua và không giải quyết vấn đề của KH

**MỨC 3 - Lỗi nặng**:
- **Lỗi kỹ năng giao tiếp**:
  + Khai thác lại thông tin lần 2, kết thúc cuộc gọi vẫn không phát hiện vấn đề của KH
  + Có cử chỉ, thái độ, ngôn ngữ hoặc hành vi không lịch sự, thiếu văn hóa, thiếu tôn trọng KH
  + Xử lý sai quy trình, tư vấn sai thông tin ảnh hưởng đến quyền lợi và việc sử dụng dịch vụ của KH
  + Không bám đuổi, không có CTA để bán hàng, bỏ quên không giải quyết vấn đề của khách hàng

---

# YÊU CẦU ĐẦU RA (BẮT BUỘC TUÂN THỦ)

**⛔ CẤM TUYỆT ĐỐI DÙNG THUẬT NGỮ KỸ THUẬT ⛔**

KHÔNG ĐƯỢC dùng các từ sau trong phần "ly_do":
- ❌ disfluency_rate, filler_count, restart_count
- ❌ spm, speed_spm, spm_std, avg_spm
- ❌ pitch_hz, pitch_std, volume_db
- ❌ hesitant_responses, len(), avg_rate
- ❌ sales_performance, sales_disfluency, sales_speed, sales_pitch
- ❌ segment (dùng "đoạn" hoặc "câu" thay thế)

**✅ PHẢI dùng ngôn ngữ tự nhiên:**
- Thay vì: "disfluency_rate = 0.46" 
  → Viết: "Sales nói không lưu loát, gần một nửa (46%) từ là từ đệm như 'dạ', 'vâng', 'ờ', 'ạ'"

- Thay vì: "speed_spm = 259" 
  → Viết: "Sales nói cực nhanh (259 từ/phút, vượt xa mức cho phép 220 từ/phút)"

- Thay vì: "len(hesitant_responses) = 2" 
  → Viết: "Có 2 lần khi khách hàng đặt câu hỏi, Sales trả lời với rất nhiều 'dạ', 'vâng', 'mà', thể hiện chưa tự tin"

- Thay vì: "segment 14 tại 69.1s (disfluency_rate = 0.333)" 
  → Viết: "Tại giây 69.1, Sales nói: '...' với 1/3 từ là từ đệm, thể hiện ngập ngừng"

```json
{{
  "chao_xung_danh": <0 hoặc 1>,
  "ky_nang_noi": <0 hoặc 1>,
  "ky_nang_nghe": <0 hoặc 1>,
  "thai_do": <0 hoặc 1>,
  "muc_loi": <"Không"|"M1"|"M2"|"M3">,
  "ly_do": "[TIÊU CHÍ 1: CHÀO/XƯNG DANH]
- Kết quả: Đạt
- Nhận xét: Có xưng danh rõ ràng trong 4 segment đầu tiên
- Bằng chứng: Tại giây 1.7s, Sales nói: 'chào anh ạ em là hương linh ở bên bizfly cloud'.

[TIÊU CHÍ 2: KỸ NĂNG NÓI]
- Kết quả: Không đạt
- Nhận xét:
  + Sales nói không lưu loát, nhiều từ đệm: Gần 50% từ là 'dạ', 'vâng', 'ờ', 'ạ'. Đây là mức ngập ngừng rất cao.
  + Tốc độ nói lên xuống: Có 2 đoạn nói quá nhanh (vượt 220 từ/phút), còn lại thì bình thường.
  + Giọng nói: Có biến thiên tốt, thể hiện nhiệt tình.
  + Trả lời khách hàng: Có 2 lần khi KH hỏi, Sales trả lời với rất nhiều từ đệm, thể hiện chưa tự tin về thông tin.
- Bằng chứng:
  + Tại giây 1.7s: 'A lô ạ a lô dạ a lô chào anh ạ...' - gần 40% từ là từ đệm.
  + Tại giây 28.9s: 'Dạ vâng bên em...' - hơn 50% từ là từ đệm.
  + Tại giây 105.6s: Sales nói cực nhanh 280 từ/phút.
  + Tại giây 162.1s: KH hỏi về 'trường hợp bị xóa', Sales trả lời: 'Dạ dạ vâng đúng rồi nếu mà mà...' - 44% từ đệm, thể hiện ngập ngừng.

[TIÊU CHÍ 3: KỸ NĂNG NGHE]
- Kết quả: Đạt
- Nhận xét: Sales lắng nghe và trả lời đầy đủ các câu hỏi của KH.

[TIÊU CHÍ 4: THÁI ĐỘ]
- Kết quả: Đạt
- Nhận xét: Thái độ lịch sự, giọng nhiệt tình, xưng hô tôn trọng 'anh ạ', 'em xin phép'.

[MỨC LỖI]
- Gán: M1
- Lý do: Sales nói không lưu loát với gần 50% từ là từ đệm, có 2 lần trả lời KH ngập ngừng nhiều thể hiện chưa tự tin, và tốc độ nói lên xuống thất thường."
}}
```

### ⚠️ Checklist BẮT BUỘC trước khi trả về:

1. ⛔ **KIỂM TRA KHÔNG CÓ THUẬT NGỮ KỸ THUẬT**:
   - Phần "ly_do" KHÔNG được chứa: disfluency_rate, filler_count, spm, speed_spm, pitch_hz, hesitant_responses, len(), segment
   - Nếu thấy thuật ngữ kỹ thuật → PHẢI viết lại bằng ngôn ngữ thông thường

2. ✅ **ĐÃ DÙNG NGÔN NGỮ TỰ NHIÊN**:
   - Viết: "Sales nói không lưu loát, 46% từ là từ đệm"
   - THAY VÌ: "disfluency_rate = 0.46"

3. 📝 **TIÊU CHÍ 2 PHẢI NGẮN GỌN**:
   - Nhận xét: Tối đa 4 điểm gạch đầu dòng (ngập ngừng, tốc độ, giọng, trả lời KH)
   - Bằng chứng: Chỉ trích 3-4 đoạn điển hình nhất
   - Mỗi đoạn trích dẫn: "Tại giây X: '...' - Y% từ đệm" (không viết dài)

4. 🎯 **MỨC LỖI PHẢI RÕ RÀNG**:
   - 1 câu ngắn gọn tóm tắt lý do (không lặp lại nhận xét trên)
   - VD: "Sales nói không lưu loát với nhiều từ đệm, có lúc trả lời KH chưa tự tin"

5. ✓ **ĐÃ TRÍCH DẪN ĐÚNG**:
   - Chỉ trích dẫn câu nói của Sales (KHÔNG trích của Customer)

6. **KIỂM TRA ĐẦU VÀO ĐÃ PHÂN LOẠI SEGMENT ĐÚNG CHƯA DỰA VÀO LỜI THOẠI(đã có trường hợp sai)**:
   - KIỂM TRA KĨ CÁC SEGMENT ĐẦU DỰA VÀO LỜI THOẠI ĐỂ XEM CÓ PHÂN BIỆT NHẦM LỜI THOẠI CỦA SALES VÀ CUSTOMS NẾU THẤY SALE KHÔNG XƯNG DANH
"""

def build_qa_prompt(call_data: dict) -> str:
    """ Xây dựng prompt chấm điểm QA bằng cách chèn dữ liệu cuộc gọi vào template. """
    call_data_str = json.dumps(call_data, indent=2, ensure_ascii=False)
    return _QA_EVALUATION_TEMPLATE.format(call_data_str=call_data_str)
