#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Giao diện Desktop App (GUI) cho mô hình OmniVoice
===============================================
Ứng dụng cho phép nhập văn bản, chọn giọng mẫu để nhân bản (Voice Cloning), 
tạo ra giọng nói và phát trực tiếp ra loa.

Chạy ứng dụng bằng lệnh:
    python app_gui.py
"""

import os
import sys
import threading
import tempfile
import logging

# Thiết lập Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Tự động kiểm tra và cài đặt thư viện thiếu
def install_and_import(package, import_name=None):
    if import_name is None:
        import_name = package
    try:
        __import__(import_name)
    except ImportError:
        import subprocess
        print(f"Không tìm thấy thư viện '{package}', đang tự động cài đặt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Cài đặt '{package}' thành công!")

install_and_import("customtkinter", "customtkinter")
install_and_import("pygame", "pygame")
install_and_import("soundfile", "soundfile")
install_and_import("librosa", "librosa")
install_and_import("transformers", "transformers")

import torch
import soundfile as sf
import customtkinter as ctk
from tkinter import filedialog, messagebox
import pygame

# Import mô hình OmniVoice
try:
    from omnivoice.models.omnivoice import OmniVoice
except ImportError:
    # Nếu không tìm thấy, thêm thư mục hiện tại vào python path
    sys.path.append(os.getcwd())
    from omnivoice.models.omnivoice import OmniVoice

# Cấu hình giao diện CustomTkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class OmniVoiceApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("OmniVoice - Trình chuyển đổi Văn bản thành Giọng nói (Local)")
        self.geometry("680x600")
        self.resizable(False, False)
        
        self.model = None
        self.temp_wav_path = None
        pygame.mixer.init()
        
        self.setup_ui()
        
        # Tự động quét và nạp checkpoint mới nhất
        self.detect_and_load_model()

    def setup_ui(self):
        # Tiêu đề ứng dụng
        self.title_label = ctk.CTkLabel(
            self, text="OMNIVOICE VIETNAMESE TTS", 
            font=ctk.CTkFont(size=22, weight="bold")
        )
        self.title_label.pack(pady=20)
        
        # Frame trạng thái nạp Model
        self.model_status_label = ctk.CTkLabel(
            self, text="Đang tìm kiếm mô hình...", 
            text_color="yellow", font=ctk.CTkFont(size=13, slant="italic")
        )
        self.model_status_label.pack(pady=5)
        
        # Khu vực nhập Text
        self.text_frame = ctk.CTkFrame(self)
        self.text_frame.pack(padx=30, pady=10, fill="x")
        
        self.input_label = ctk.CTkLabel(
            self.text_frame, text="Nhập văn bản tiếng Việt cần tạo giọng nói:", 
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.input_label.pack(anchor="w", padx=15, pady=5)
        
        self.text_input = ctk.CTkTextbox(self.text_frame, height=120)
        self.text_input.pack(fill="x", padx=15, pady=10)
        self.text_input.insert("1.0", "Xin chào bạn! Đây là giọng nói nhân bản được tạo hoàn toàn bằng AI.")
        
        # Khu vực cấu hình nhân bản giọng nói (Voice Cloning)
        self.clone_frame = ctk.CTkFrame(self)
        self.clone_frame.pack(padx=30, pady=10, fill="x")
        
        self.clone_label = ctk.CTkLabel(
            self.clone_frame, text="Tùy chọn nhân bản giọng nói (Voice Cloning):", 
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.clone_label.grid(row=0, column=0, columnspan=2, sticky="w", padx=15, pady=5)
        
        self.ref_audio_btn = ctk.CTkButton(
            self.clone_frame, text="Chọn file giọng mẫu (.wav)", 
            command=self.select_ref_audio, width=180
        )
        self.ref_audio_btn.grid(row=1, column=0, padx=15, pady=10)
        
        self.ref_audio_path_label = ctk.CTkLabel(
            self.clone_frame, text="Không có giọng mẫu (Sử dụng giọng mặc định)", 
            text_color="gray", font=ctk.CTkFont(size=12)
        )
        self.ref_audio_path_label.grid(row=1, column=1, sticky="w", padx=10)
        
        self.ref_text_label = ctk.CTkLabel(
            self.clone_frame, text="Nhập nội dung văn bản của file giọng mẫu trên:"
        )
        self.ref_text_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=15, pady=2)
        
        self.ref_text_input = ctk.CTkEntry(
            self.clone_frame, placeholder_text="Lời thoại của file âm thanh mẫu (bắt buộc nếu chọn file mẫu)", 
            width=500
        )
        self.ref_text_input.grid(row=3, column=0, columnspan=2, padx=15, pady=5, sticky="we")
        
        self.ref_audio_path = None

        # Nút điều khiển
        self.ctrl_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.ctrl_frame.pack(pady=20)
        
        self.gen_btn = ctk.CTkButton(
            self.ctrl_frame, text="TẠO GIỌNG NÓI (TTS)", 
            command=self.start_generation, fg_color="#1f538d", 
            hover_color="#14375e", width=200, height=45, 
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.gen_btn.grid(row=0, column=0, padx=15)
        
        self.play_btn = ctk.CTkButton(
            self.ctrl_frame, text="PHÁT LOA", 
            command=self.play_audio, fg_color="#2b712c", 
            hover_color="#1b471c", width=150, height=45,
            font=ctk.CTkFont(size=14, weight="bold"), state="disabled"
        )
        self.play_btn.grid(row=0, column=1, padx=15)
        
        # Thanh trạng thái dưới cùng
        self.status_bar = ctk.CTkLabel(
            self, text="Sẵn sàng.", 
            anchor="w", fg_color="#202020", padx=10, height=30
        )
        self.status_bar.pack(side="bottom", fill="x")

    def detect_and_load_model(self):
        # Quét checkpoint ở cả hai nơi: exp/ và exp/omnivoice_finetune/
        checkpoints = []
        for path in ["exp/omnivoice_finetune", "exp"]:
            if os.path.exists(path):
                checkpoints.extend(glob.glob(os.path.join(path, "checkpoint-*")))
                
        # Lọc ra các thư mục hợp lệ
        checkpoints = [c for c in checkpoints if os.path.isdir(c)]
        
        if checkpoints:
            # Sắp xếp để tìm checkpoint mới nhất theo số step
            checkpoints.sort(key=lambda x: int(os.path.basename(x).split("-")[-1]))
            latest_ckpt = checkpoints[-1]
            self.model_status_label.configure(
                text=f"Phát hiện checkpoint: {os.path.basename(latest_ckpt)}. Đang nạp...",
                text_color="lightblue"
            )
            # Chạy luồng phụ để nạp model không bị đơ giao diện
            threading.Thread(target=self.load_model_worker, args=(latest_ckpt,), daemon=True).start()
            return
        
        # Nếu không có checkpoint nào, sử dụng model mặc định từ HuggingFace
        self.model_status_label.configure(
            text="Không tìm thấy checkpoint. Sẽ tải và nạp mô hình gốc 'k2-fsa/OmniVoice' từ HF...",
            text_color="orange"
        )
        threading.Thread(target=self.load_model_worker, args=("k2-fsa/OmniVoice",), daemon=True).start()

    def load_model_worker(self, model_path):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(f"Loading model {model_path} on {device}...")
            
            # Tự động chọn kiểu dữ liệu phù hợp
            dtype = torch.float16 if device == "cuda" else torch.float32
            
            self.model = OmniVoice.from_pretrained(
                model_path, device_map=device, dtype=dtype
            )
            self.model_status_label.configure(
                text=f"Đã nạp thành công: {os.path.basename(model_path)} (Chạy trên {device.upper()})",
                text_color="green"
            )
            logging.info("Model loaded successfully.")
        except Exception as e:
            self.model_status_label.configure(
                text=f"Lỗi khi nạp mô hình: {str(e)}",
                text_color="red"
            )
            logging.error(f"Failed to load model: {e}")

    def select_ref_audio(self):
        file_path = filedialog.askopenfilename(
            title="Chọn file giọng nói mẫu",
            filetypes=[("Audio files", "*.wav")]
        )
        if file_path:
            self.ref_audio_path = file_path
            self.ref_audio_path_label.configure(
                text=os.path.basename(file_path),
                text_color="lightgreen"
            )
            # Tìm xem có file text đi kèm trùng tên không
            txt_path = os.path.splitext(file_path)[0] + ".txt"
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    self.ref_text_input.delete(0, "end")
                    self.ref_text_input.insert(0, f.read().strip())
        else:
            self.ref_audio_path = None
            self.ref_audio_path_label.configure(
                text="Không có giọng mẫu (Sử dụng giọng mặc định)",
                text_color="gray"
            )

    def start_generation(self):
        if self.model is None:
            messagebox.showerror("Lỗi", "Vui lòng đợi mô hình nạp xong!")
            return
            
        text = self.text_input.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập văn bản cần phát âm!")
            return
            
        # Kiểm tra điều kiện nhân bản giọng nói
        ref_text = self.ref_text_input.get().strip()
        if self.ref_audio_path and not ref_text:
            messagebox.showwarning("Cảnh báo", "Bạn đã chọn file giọng mẫu nhưng chưa nhập nội dung lời thoại tương ứng!")
            return

        self.gen_btn.configure(state="disabled", text="ĐANG XỬ LÝ...")
        self.play_btn.configure(state="disabled")
        self.status_bar.configure(text="Đang sinh giọng nói...")
        
        # Chạy suy luận trên luồng phụ để tránh đơ app
        threading.Thread(target=self.generate_worker, args=(text, ref_text), daemon=True).start()

    def generate_worker(self, text, ref_text):
        try:
            logging.info(f"Generating audio for text: {text}")
            
            # Thực thi sinh giọng nói
            audios = self.model.generate(
                text=text,
                language="vi",
                ref_audio=self.ref_audio_path,
                ref_text=ref_text if self.ref_audio_path else None,
                num_step=32,
                guidance_scale=2.0
            )
            
            # Tạo file tạm thời để lưu âm thanh đầu ra
            if self.temp_wav_path and os.path.exists(self.temp_wav_path):
                try:
                    pygame.mixer.music.unload()
                    os.remove(self.temp_wav_path)
                except Exception:
                    pass
                    
            fd, self.temp_wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            
            # Ghi dữ liệu vào file wav tạm để phát loa
            sf.write(self.temp_wav_path, audios[0], self.model.sampling_rate)
            logging.info(f"Saved generated audio to: {self.temp_wav_path}")
            
            # Tạo thư mục output nếu chưa có
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"output_{timestamp}.wav"
            
            latest_path = os.path.join(output_dir, "latest.wav")
            history_path = os.path.join(output_dir, filename)
            
            try:
                # Đóng nhạc pygame trước để tránh lỗi lock file nếu ghi đè
                pygame.mixer.music.unload()
                
                # Lưu file mới nhất và file lịch sử
                sf.write(latest_path, audios[0], self.model.sampling_rate)
                sf.write(history_path, audios[0], self.model.sampling_rate)
                
                logging.info(f"Saved copy to: {os.path.abspath(latest_path)} and {os.path.abspath(history_path)}")
            except Exception as pe:
                logging.warning(f"Không thể lưu file vào thư mục output: {pe}")
            
            # Cập nhật giao diện
            self.gen_btn.configure(state="normal", text="TẠO GIỌNG NÓI (TTS)")
            self.play_btn.configure(state="normal")
            self.status_bar.configure(text=f"Tạo xong! Lưu tại: {output_dir}/{filename}")
            
            # Tự động phát âm thanh ngay sau khi sinh xong
            self.play_audio()
            
        except Exception as e:
            logging.error(f"Error during generation: {e}")
            self.gen_btn.configure(state="normal", text="TẠO GIỌNG NÓI (TTS)")
            self.status_bar.configure(text=f"Lỗi: {str(e)}")
            messagebox.showerror("Lỗi", f"Quá trình sinh âm thanh gặp lỗi:\n{str(e)}")

    def play_audio(self):
        if self.temp_wav_path and os.path.exists(self.temp_wav_path):
            try:
                pygame.mixer.music.load(self.temp_wav_path)
                pygame.mixer.music.play()
                self.status_bar.configure(text="Đang phát âm thanh...")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể phát âm thanh: {str(e)}")
                self.status_bar.configure(text=f"Lỗi phát nhạc: {str(e)}")

if __name__ == "__main__":
    # Import glob để quét file
    import glob
    app = OmniVoiceApp()
    app.mainloop()
