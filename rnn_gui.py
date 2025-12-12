import customtkinter as ctk
import threading
import time

from rnn_backend import SimpleRNN 

class RNNApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("RNN Text Generator Pro (Modular Version)")
        self.geometry("900x650")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.rnn = SimpleRNN() 

        # LAYOUT
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 1. Sidebar
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="AI CONFIG", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Inputs Sidebar
        self.label_epoch = ctk.CTkLabel(self.sidebar_frame, text="Số Epochs:", anchor="w")
        self.label_epoch.grid(row=1, column=0, padx=20, pady=(10, 0))
        self.entry_epoch = ctk.CTkEntry(self.sidebar_frame)
        self.entry_epoch.grid(row=2, column=0, padx=20, pady=(0, 10))
        self.entry_epoch.insert(0, "2000")

        self.label_lr = ctk.CTkLabel(self.sidebar_frame, text="Learning Rate:", anchor="w")
        self.label_lr.grid(row=3, column=0, padx=20, pady=(10, 0))
        self.entry_lr = ctk.CTkEntry(self.sidebar_frame)
        self.entry_lr.grid(row=4, column=0, padx=20, pady=(0, 10))
        self.entry_lr.insert(0, "0.1")

        self.label_hidden = ctk.CTkLabel(self.sidebar_frame, text="Hidden Size:", anchor="w")
        self.label_hidden.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.entry_hidden = ctk.CTkEntry(self.sidebar_frame)
        self.entry_hidden.grid(row=6, column=0, padx=20, pady=(0, 20), sticky="n")
        self.entry_hidden.insert(0, "32")

        # 2. Main Area
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(4, weight=1)

        self.header = ctk.CTkLabel(self.main_frame, text="HUẤN LUYỆN MÔ HÌNH RNN", font=ctk.CTkFont(size=24, weight="bold"))
        self.header.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        self.label_input = ctk.CTkLabel(self.main_frame, text="Dữ liệu mẫu (Training Data):", anchor="w")
        self.label_input.grid(row=1, column=0, padx=20, pady=(5, 0), sticky="w")
        
        self.textbox_input = ctk.CTkTextbox(self.main_frame, height=80)
        self.textbox_input.grid(row=2, column=0, padx=20, pady=(5, 10), sticky="ew")
        self.textbox_input.insert("0.0", "hello world engineering") 

        self.train_btn = ctk.CTkButton(self.main_frame, text="Bắt đầu Training", command=self.start_training_thread, height=40, fg_color="#2CC985", hover_color="#229A65")
        self.train_btn.grid(row=3, column=0, padx=20, pady=10, sticky="ew")

        self.progress_bar = ctk.CTkProgressBar(self.main_frame)
        self.progress_bar.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        self.progress_bar.set(0)

        self.log_box = ctk.CTkTextbox(self.main_frame, height=150, state="disabled", fg_color="#1D1E1E")
        self.log_box.grid(row=5, column=0, padx=20, pady=10, sticky="nsew")

        # Testing Area
        self.test_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.test_frame.grid(row=6, column=0, padx=20, pady=20, sticky="ew")
        self.test_frame.grid_columnconfigure(1, weight=1)

        self.label_test = ctk.CTkLabel(self.test_frame, text="Ký tự bắt đầu:", anchor="w")
        self.label_test.grid(row=0, column=0, padx=(0, 10))

        self.entry_char = ctk.CTkEntry(self.test_frame, width=50, placeholder_text="h")
        self.entry_char.grid(row=0, column=1, padx=(0, 10), sticky="w")

        self.test_btn = ctk.CTkButton(self.test_frame, text="Dự đoán", command=self.generate_text, fg_color="#3B8ED0")
        self.test_btn.grid(row=0, column=2, padx=10)

        self.lbl_result = ctk.CTkLabel(self.test_frame, text="Kết quả: ...", font=ctk.CTkFont(size=16, weight="bold"), text_color="#FFD700")
        self.lbl_result.grid(row=1, column=0, columnspan=3, pady=(15, 0), sticky="w")


    def log(self, message):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def update_progress(self, val):
        self.progress_bar.set(val)

    def start_training_thread(self):
        raw_text = self.textbox_input.get("0.0", "end").strip()
        if len(raw_text) < 2:
            self.log("LỖI: Văn bản quá ngắn!")
            return

        try:
            epochs = int(self.entry_epoch.get())
            lr = float(self.entry_lr.get())
            hidden = int(self.entry_hidden.get())
        except ValueError:
            self.log("LỖI: Nhập số liệu sai!")
            return

        self.rnn.hidden_size = hidden
        
        self.train_btn.configure(state="disabled", text="Đang Training...")
        self.log_box.configure(state="normal")
        self.log_box.delete("0.0", "end")
        self.log_box.configure(state="disabled")

        threading.Thread(target=self.run_training, args=(raw_text, epochs, lr)).start()

    def run_training(self, text, epochs, lr):
        self.log(f"Bắt đầu học: '{text}'")
        start_time = time.time()

        self.rnn.train(text, epochs, lr, self.update_progress, self.log)
        
        elapsed = time.time() - start_time
        self.log(f"Thời gian: {elapsed:.2f}s")
        self.train_btn.configure(state="normal", text="Bắt đầu Training")

    def generate_text(self):
        start_char = self.entry_char.get().strip()
        if not start_char:
            full_text = self.textbox_input.get("0.0", "end").strip()
            if full_text: start_char = full_text[0]
            else: return

        target_len = len(self.textbox_input.get("0.0", "end").strip()) - 1
        result = self.rnn.predict(start_char[0], target_len)
        self.lbl_result.configure(text=f"Kết quả: {result}")

if __name__ == "__main__":
    app = RNNApp()
    app.mainloop()