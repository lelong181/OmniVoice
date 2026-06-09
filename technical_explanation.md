# BẢN PHÂN TÍCH CHUYÊN MÔN KỸ THUẬT & KIẾN TRÚC MÃ NGUỒN OMNIVOICE TTS

Báo cáo này cung cấp thông tin chuyên sâu về mặt kỹ thuật, giải thuật và cấu trúc lập trình của dự án **Vietnamese OmniVoice TTS** dựa trên các mã nguồn [app_gui.py](file:///d:/src/NPL/OmniVoice/app_gui.py), [trainer.py](file:///d:/src/NPL/OmniVoice/omnivoice/training/trainer.py), và tài liệu nghiên cứu của hệ thống.

---

## 1. Kiến Trúc Mô Hình Học Máy (Model Architecture)

Hệ thống OmniVoice hoạt động theo trường phái **LLM-based Text-to-Speech (TTS)**. Thay vì sử dụng mô hình hai giai đoạn cổ điển (Text ➔ Mel-Spectrogram ➔ Vocoder ➔ Sóng âm), OmniVoice tối ưu hóa quy trình thông qua cơ chế tự hồi quy (autoregressive) trên không gian token âm thanh rời rạc.

* **Trang bị Backbone Qwen3-0.6B**: Một mô hình ngôn ngữ lớn (LLM) thu nhỏ với 600 triệu tham số, đóng vai trò dự đoán mã âm thanh tiếp theo dựa trên chuỗi văn bản tiếng Việt đầu vào. Do kích thước tối giản, mô hình cho phép chạy suy luận thời gian thực (real-time inference) ngay trên các GPU phổ thông như GTX 1650.
* **Higgs Audio V2 Tokenizer**: Sử dụng cơ chế lượng hóa vector dư (**Residual Vector Quantization - RVQ**). Sóng âm thanh 24kHz liên tục được chuyển hóa thành các chỉ mục số (indices) đại diện cho các đặc trưng tiềm ẩn của âm thanh.

---

## 2. Giải Thuật Tiền Xử Lý & Lượng Hóa Dữ Liệu (Data Pipeline)

Luồng xử lý dữ liệu từ tệp âm thanh thô `.wav` đầu vào đến định dạng tập dữ liệu phân tán (WebDataset) trải qua 3 giai đoạn chính:

```
[Audio Gốc 24kHz] 
       │
       ▼ (Sử dụng OpenAI Whisper Base để cắt mốc câu & phiên âm văn bản)
[823 Audio Chunks & Lời thoại tương ứng] 
       │
       ▼ (Đưa qua Higgs Audio V2 Tokenizer thực hiện RVQ)
[8 Lớp Audio Tokens rời rạc (Codebooks)] 
       │
       ▼ (Đóng gói dữ liệu văn bản + token thành các file .tar)
[WebDataset Shards]
```

### Residual Vector Quantization (RVQ)
Cơ chế RVQ phân tách thông tin âm thanh thành **8 lớp (8 codebooks)** hoạt động song song. Mỗi lớp ánh xạ đặc trưng âm thanh tiềm ẩn vào một không gian từ điển có kích thước 1024 token (`0` đến `1023` và 1 token mask):
* **Lớp 1 - 2**: Lưu trữ thông tin thô căn bản (tông giọng chính, cao độ, ngữ điệu, âm tố cơ bản).
* **Lớp 3 - 6**: Lưu trữ chi tiết trung tần (độ rõ chữ, nét đặc trưng giọng nói của người nói).
* **Lớp 7 - 8**: Lưu trữ chi tiết cao tần (nhạc tính, độ vang và các chi tiết âm tần số cao).

---

## 3. Quy Trình Huấn Luyện & Giải Pháp Tối Ưu VRAM (Training Optimizations)

Các thuật toán tối ưu hóa cốt lõi được triển khai trong tệp [trainer.py](file:///d:/src/NPL/OmniVoice/omnivoice/training/trainer.py):

### Hàm Mất Mát Cross-Entropy Phân Cấp (Hierarchical Loss)
Hàm mục tiêu sử dụng hàm mất mát Cross-Entropy tính toán song song trên cả 8 lớp lượng hóa âm thanh với bộ trọng số phân rã:
$$\mathcal{L} = \sum_{c=1}^{8} w_c \cdot \text{CrossEntropy}(P(y_{t,c} | y_{<t}, X_{\text{text}}), \hat{y}_{t,c})$$
Trong đó $w_c = [8, 8, 6, 6, 4, 4, 2, 2]$ tương ứng với các lớp từ 1 đến 8.
* **Mục đích**: Buộc mạng nơ-ron phải học cách tái tạo cấu trúc âm thanh cốt lõi và phát âm chính xác tiếng Việt (đầu ra của codebook 1 và 2 có trọng số phạt lỗi lớn nhất là `8`) trước khi học các chi tiết phụ trợ ở tần số cao.

### Thuật Toán Adafactor Tiết Kiệm Bộ Nhớ
Để giải quyết bài toán tràn bộ nhớ GPU (OOM) trên các phần cứng giới hạn (GPU T4 15GB hoặc GTX 1650 4GB), hệ thống đã thay thế trình tối ưu hóa AdamW bằng **Adafactor** tại phương thức [create_optimizer_and_scheduler](file:///d:/src/NPL/OmniVoice/omnivoice/training/trainer.py#L161-L196):
* **AdamW**: Yêu cầu lưu trữ các trạng thái mô-men động lượng bậc một và bậc hai riêng biệt cho từng tham số của mô hình $\theta$. Bộ nhớ yêu cầu bổ sung xấp xỉ:
  $$\text{Memory}_{\text{AdamW State}} \approx 8 \times N_{\text{params}} \text{ bytes}$$
  Với mô hình 600M tham số, AdamW tiêu tốn thêm khoảng **4.8 GB VRAM** chỉ để lưu trữ trạng thái tối ưu.
* **Adafactor**: Sử dụng kỹ thuật phân rã ma trận gradient thành tích Kronecker của các vector dòng và cột. Bộ nhớ lưu trữ trạng thái giảm từ bậc nhân $\mathcal{O}(d_1 \times d_2)$ xuống bậc cộng $\mathcal{O}(d_1 + d_2)$ (Adafactor Memory Factor):
  $$\text{Memory}_{\text{Adafactor State}} \approx 4 \times (d_1 + d_2) \text{ bytes}$$
* **Kết quả**: Tiết kiệm thực tế khoảng **5.1 GB VRAM**, đưa tổng lượng VRAM tiêu thụ đỉnh của cả mô hình khi train xuống mức an toàn **9.51 GB**.

---

## 4. Tự Động Hóa MLOps Trực Tuyến (CI/CD Kaggle Pipeline)

Quy trình MLOps khép kín được tự động hóa thông qua công cụ API của Kaggle:

1. **Kiểm soát chất lượng (Quality Gate)**: GitHub Actions chạy thử nghiệm tích hợp (`Smoke Test` 2 steps) trên môi trường CPU nhằm đảm bảo mã nguồn không chứa lỗi cú pháp hoặc lỗi logic mạng nơ-ron trước khi kích hoạt tài nguyên GPU đám mây.
2. **Kích hoạt tự động (Kaggle API)**: Sử dụng thư viện `kaggle` và thông tin credentials từ GitHub Secrets để đóng gói mã nguồn mới nhất và đẩy lên Kaggle dưới dạng một Script Kernel qua tệp [run_kaggle.py](file:///d:/src/NPL/OmniVoice/run_kaggle.py).
3. **Quản lý ổ cứng & Auto-Resume**:
   * Mỗi checkpoint của mô hình nặng tới **2.45 GB**. Để tránh vượt quá giới hạn đĩa cứng 20GB của Kaggle, hàm [save_checkpoint](file:///d:/src/NPL/OmniVoice/omnivoice/training/checkpoint.py) chỉ duy trì tối đa 2 checkpoint mới nhất (`keep_last_n_checkpoints: 2`).
   * Cơ chế `"resume_from_checkpoint": "latest"` tự động quét thư mục `exp/` để khôi phục trạng thái huấn luyện từ checkpoint có chỉ số bước (`global_step`) lớn nhất nếu phiên chạy trên Kaggle bị ngắt quãng nửa chừng.

---

## 5. Thiết Kế Mã Nguồn Giao Diện Desktop (Local GUI)

Ứng dụng Desktop chạy cục bộ được viết bằng thư viện `CustomTkinter` trong tệp [app_gui.py](file:///d:/src/NPL/OmniVoice/app_gui.py), áp dụng các kỹ thuật xử lý bất đồng bộ:

### Xử Lý Đa Luồng Tránh Treo GUI (Multithreading)
Các tác vụ có thời gian thực thi dài được đẩy vào luồng phụ (`threading.Thread`) để đảm bảo luồng giao diện chính (Main GUI Thread) luôn phản hồi mượt mà:
* **Nạp mô hình**: Phương thức [load_model_worker](file:///d:/src/NPL/OmniVoice/app_gui.py#L243) thực hiện nạp checkpoint từ đĩa vào RAM/VRAM trong luồng phụ. Giao diện hiển thị trạng thái động dạng `"Đang nạp..."` để thông báo cho người dùng.
* **Tạo giọng nói (Inference)**: Phương thức [generate_worker](file:///d:/src/NPL/OmniVoice/app_gui.py#L324) gọi phương thức sinh âm thanh của mô hình `self.model.generate` trong luồng phụ.

### Quản Lý Bộ Nhớ Khi Chuyển Thiết Bị (Device Management)
Khi người dùng chuyển đổi thiết bị chạy từ GPU (CUDA) sang CPU hoặc ngược lại thông qua phương thức [change_device](file:///d:/src/NPL/OmniVoice/app_gui.py#L190):
1. Giải phóng tham chiếu mô hình cũ: `self.model = None`.
2. Gọi bộ gom rác của Python để thu hồi bộ nhớ Heap: `gc.collect()`.
3. Giải phóng bộ nhớ đệm đã cấp phát trên GPU: `torch.cuda.empty_cache()`.
4. Bắt đầu luồng phụ để nạp mô hình mới lên thiết bị đích.

### Trình Quản Lý Phát & Lưu Trữ Âm Thanh
* Sử dụng thư viện `pygame.mixer` để điều khiển thiết bị âm thanh đầu ra. Trước khi thực hiện ghi đè hoặc tạo mới âm thanh, ứng dụng sẽ gọi `pygame.mixer.music.unload()` để giải phóng quyền khóa tệp tin âm thanh tạm thời trên hệ điều hành Windows.
* Tự động xuất tệp tin âm thanh ra thư mục `output/` dưới hai định dạng: tệp tin mới nhất [latest.wav](file:///d:/src/NPL/OmniVoice/output/latest.wav) và tệp tin lịch sử đính kèm dấu thời gian `output_YYYYMMDD_HHMMSS.wav` để lưu trữ dữ liệu.
