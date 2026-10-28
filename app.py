import os
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
from main import BarPathAnalyzer


class BarPathStudioApp:
    """
    BarPath AI Studio - Tüm Kontrolleri Canlı ve Anlık Çalışan (25 FPS) Masaüstü Arayüzü.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("BarPath AI Studio - Canlı Hız (VBT) & Form Analiz Sistemi")
        self.root.geometry("1360x800")
        self.root.minsize(1050, 680)
        self.root.configure(bg="#0f1117")

        self.video_path = "video.mp4" if os.path.exists("video.mp4") else ""
        self.model_path = "best.pt" if os.path.exists("best.pt") else ""

        self.cap = None
        self.analyzer = None
        self.is_playing = False
        self.frame_idx = 0
        self.total_frames = 690
        self.fps = 25.0
        self.manual_click_mode = False
        self.last_raw_frame = None
        self._canvas_img_id = None
        self._display_rect = (0, 0, 1280, 720)  # (ox, oy, draw_w, draw_h)

        self._build_ui()
        # Pencere boyutlandıktan hemen sonra canlı analizi otomatik başlat
        self.root.after(200, self._start_live_analysis)

    def _build_ui(self):
        # Üst Başlık Barı
        header = tk.Frame(self.root, bg="#161922", height=56)
        header.pack(fill=tk.X, side=tk.TOP)

        tk.Label(
            header,
            text="🏋️‍♂️ BARPATH AI STUDIO",
            font=("Segoe UI", 15, "bold"),
            fg="#4ade80",
            bg="#161922",
            padx=18,
            pady=10,
        ).pack(side=tk.LEFT)

        tk.Label(
            header,
            text="Canlı Halter Bar Yolu (Bar Path), Hız (VBT) ve Form Analiz Stüdyosu",
            font=("Segoe UI", 10),
            fg="#94a3b8",
            bg="#161922",
        ).pack(side=tk.LEFT, padx=6)

        # Ana Gövde
        main_frame = tk.Frame(self.root, bg="#0f1117")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)

        # Sol Kontrol Paneli (Sabit genişlik: 330 px)
        sidebar = tk.Frame(main_frame, bg="#161922", width=330, padx=14, pady=12)
        sidebar.pack(fill=tk.Y, side=tk.LEFT, padx=(0, 12))
        sidebar.pack_propagate(False)

        self._section_label(sidebar, "1. VİDEO SEÇİMİ")
        tk.Button(
            sidebar,
            text="📂 Farklı Antrenman Videosu Seç",
            command=self._select_video,
            bg="#2563eb",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            pady=6,
            cursor="hand2",
        ).pack(fill=tk.X, pady=(2, 10))

        self._section_label(sidebar, "2. CANLI TAKİP NOKTASI (ANINDA DEĞİŞİR)")
        self.track_mode_var = tk.StringVar(value="right_plate")
        modes = [
            ("Sağ Plaka Göbeği (Önerilen)", "right_plate"),
            ("Bar Tam Ortası (Merkez)", "center"),
            ("Sol Plaka Göbeği", "left_plate"),
        ]
        for text, val in modes:
            rb = tk.Radiobutton(
                sidebar,
                text=text,
                variable=self.track_mode_var,
                value=val,
                command=self._on_mode_change,
                bg="#161922",
                fg="#e2e8f0",
                selectcolor="#0f1117",
                activebackground="#161922",
                activeforeground="#4ade80",
                font=("Segoe UI", 9),
                cursor="hand2",
            )
            rb.pack(anchor=tk.W, pady=2)

        self.btn_manual = tk.Button(
            sidebar,
            text="🎯 Videodan Plakayı Elle Seç (Tıkla-Takip Et)",
            command=self._enable_manual_click,
            bg="#334155",
            fg="#f8fafc",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            pady=6,
            cursor="hand2",
        )
        self.btn_manual.pack(fill=tk.X, pady=(6, 10))

        self._section_label(sidebar, "3. ÇİZGİ İZİ UZUNLUĞU (KARELER)")
        self.trail_scale = tk.Scale(
            sidebar,
            from_=25,
            to=450,
            orient=tk.HORIZONTAL,
            command=self._on_trail_change,
            bg="#161922",
            fg="#4ade80",
            highlightthickness=0,
            troughcolor="#0f1117",
        )
        self.trail_scale.set(180)
        self.trail_scale.pack(fill=tk.X, pady=(0, 8))

        self._section_label(sidebar, "4. ÇİZGİ KALINLIĞI (PX)")
        self.thick_scale = tk.Scale(
            sidebar,
            from_=2,
            to=10,
            orient=tk.HORIZONTAL,
            command=self._on_thickness_change,
            bg="#161922",
            fg="#38bdf8",
            highlightthickness=0,
            troughcolor="#0f1117",
        )
        self.thick_scale.set(4)
        self.thick_scale.pack(fill=tk.X, pady=(0, 8))

        self._section_label(sidebar, "5. ZORLANMA RENK HASSASİYETİ (m/s)")
        self.vel_scale = tk.Scale(
            sidebar,
            from_=0.20,
            to=0.55,
            resolution=0.02,
            orient=tk.HORIZONTAL,
            command=self._on_vel_change,
            bg="#161922",
            fg="#facc15",
            highlightthickness=0,
            troughcolor="#0f1117",
        )
        self.vel_scale.set(0.36)
        self.vel_scale.pack(fill=tk.X, pady=(0, 10))

        self._section_label(sidebar, "6. OYNATMA KONTROLLERİ")
        ctrl_row = tk.Frame(sidebar, bg="#161922")
        ctrl_row.pack(fill=tk.X, pady=4)

        self.btn_pause = tk.Button(
            ctrl_row,
            text="⏸ Duraklat",
            command=self._toggle_pause,
            bg="#16a34a",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            pady=7,
            cursor="hand2",
        )
        self.btn_pause.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        tk.Button(
            ctrl_row,
            text="🔄 Baştan Başlat",
            command=self._start_live_analysis,
            bg="#ea580c",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            pady=7,
            cursor="hand2",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        self._section_label(sidebar, "RENK REHBERİ")
        legend_txt = (
            "🟢 YEŞİL: Hızlı & dikey hatta sadık form\n"
            "🟡 SARI: Yavaşlama veya hafif yatay sapma\n"
            "🔴 KIRMIZI: Sticking Point (Zorlanma/Sapma)"
        )
        tk.Label(
            sidebar,
            text=legend_txt,
            justify=tk.LEFT,
            bg="#0f1117",
            fg="#cbd5e1",
            font=("Segoe UI", 8),
            padx=8,
            pady=8,
        ).pack(fill=tk.X, pady=4)

        self.status_var = tk.StringVar(value="Canlı analiz başlatılıyor...")
        tk.Label(
            sidebar,
            textvariable=self.status_var,
            wraplength=295,
            justify=tk.LEFT,
            bg="#161922",
            fg="#38bdf8",
            font=("Segoe UI", 9, "bold"),
        ).pack(side=tk.BOTTOM, anchor=tk.W, pady=4)

        # Sağ Video Görüntüleme Alanı (Tam sığan Canvas + İlerleme Çubuğu)
        right_panel = tk.Frame(main_frame, bg="#161922", padx=10, pady=10)
        right_panel.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        right_panel.pack_propagate(False)

        self.video_canvas = tk.Canvas(
            right_panel, bg="#090b10", highlightthickness=0, cursor="crosshair"
        )
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        self.video_canvas.bind("<Button-1>", self._on_video_click)
        self.video_canvas.bind("<Configure>", self._on_canvas_resize)

        # Alt Zaman Çubuğu (Timeline)
        bottom_bar = tk.Frame(right_panel, bg="#161922", height=32)
        bottom_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(6, 0))

        self.frame_info_var = tk.StringVar(value="Kare: 0 / 690")
        tk.Label(
            bottom_bar,
            textvariable=self.frame_info_var,
            bg="#161922",
            fg="#94a3b8",
            font=("Segoe UI", 9),
        ).pack(side=tk.LEFT)

        tk.Label(
            bottom_bar,
            text="💡 İpucu: Video oynarken sol paneldeki ayarları değiştirebilir veya doğrudan videoda istediğiniz noktaya tıklayabilirsiniz!",
            bg="#161922",
            fg="#4ade80",
            font=("Segoe UI", 9),
        ).pack(side=tk.RIGHT)

    @staticmethod
    def _section_label(parent, text):
        tk.Label(
            parent,
            text=text,
            font=("Segoe UI", 8, "bold"),
            fg="#94a3b8",
            bg="#161922",
        ).pack(anchor=tk.W, pady=(6, 2))

    # --- CANLI KONTROL GERİ ÇAĞIRMALARI (CALLBACKS) ---
    def _on_mode_change(self):
        mode = self.track_mode_var.get()
        self.manual_click_mode = False
        self.btn_manual.configure(bg="#334155", text="🎯 Videodan Plakayı Elle Seç (Tıkla-Takip Et)")
        if self.analyzer is not None:
            self.analyzer.set_track_mode(mode)
            names = {
                "right_plate": "Sağ Plaka Göbeği",
                "center": "Bar Tam Ortası (Merkez)",
                "left_plate": "Sol Plaka Göbeği",
            }
            self.status_var.set(f"✅ Takip noktası anında '{names.get(mode, mode)}' olarak değiştirildi!")
            self._refresh_paused_frame()

    def _on_trail_change(self, val):
        if self.analyzer is not None:
            self.analyzer.trail_length = int(float(val))
            self.status_var.set(f"📏 Çizgi izi uzunluğu: {int(float(val))} kare")
            self._refresh_paused_frame()

    def _on_thickness_change(self, val):
        if self.analyzer is not None:
            self.analyzer.line_thickness = int(float(val))
            self.status_var.set(f"🖌️ Çizgi kalınlığı: {int(float(val))} px")
            self._refresh_paused_frame()

    def _on_vel_change(self, val):
        if self.analyzer is not None:
            self.analyzer.vel_threshold = float(val)
            self.status_var.set(f"⚡ Zorlanma hız eşiği: {float(val):.2f} m/s")
            self._refresh_paused_frame()

    def _enable_manual_click(self):
        self.manual_click_mode = True
        self.btn_manual.configure(bg="#dc2626", text="🎯 ŞİMDİ VİDEODA PLAKAYA TIKLAYIN!")
        self.status_var.set("🎯 Sağdaki videoda takip etmek istediğiniz plakanın veya barın üzerine tıklayın!")

    def _on_video_click(self, event):
        if self.analyzer is None:
            return
        ox, oy, dw, dh = self._display_rect
        if dw <= 0 or dh <= 0:
            return
        # Canvas koordinatını 1280x720 video koordinatına çevir
        vx = (event.x - ox) * (self.analyzer.target_width / dw)
        vy = (event.y - oy) * (self.analyzer.target_height / dh)
        if not (0 <= vx <= self.analyzer.target_width and 0 <= vy <= self.analyzer.target_height):
            return

        self.manual_click_mode = False
        self.btn_manual.configure(bg="#334155", text="🎯 Videodan Plakayı Elle Seç (Tıkla-Takip Et)")
        self.analyzer.set_manual_point(vx, vy)
        self.status_var.set(f"🎯 Manuel kilit aktif! ({int(vx)}, {int(vy)}) noktası takip ediliyor.")

        if self.last_raw_frame is not None:
            annotated = self.analyzer.render_current_overlay(self.last_raw_frame)
            self._display_cv_frame(annotated)

    def _on_canvas_resize(self, _event):
        if not self.is_playing and self.last_raw_frame is not None and self.analyzer is not None:
            annotated = self.analyzer.render_current_overlay(self.last_raw_frame)
            self._display_cv_frame(annotated)

    def _refresh_paused_frame(self):
        if not self.is_playing and self.last_raw_frame is not None and self.analyzer is not None:
            annotated = self.analyzer.render_current_overlay(self.last_raw_frame)
            self._display_cv_frame(annotated)

    def _select_video(self):
        path = filedialog.askopenfilename(
            title="Antrenman Videosu Seç",
            filetypes=[("Video Dosyaları", "*.mp4 *.avi *.mov *.mkv")],
        )
        if path:
            self.video_path = path
            self.status_var.set(f"📂 Seçilen video: {os.path.basename(path)}")
            self._start_live_analysis()

    def _display_cv_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        cw = max(self.video_canvas.winfo_width(), 640)
        ch = max(self.video_canvas.winfo_height(), 360)

        scale = min(cw / w, ch / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        ox = (cw - new_w) // 2
        oy = (ch - new_h) // 2
        self._display_rect = (ox, oy, new_w, new_h)

        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = Image.fromarray(resized)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_canvas.imgtk = imgtk

        if self._canvas_img_id is None:
            self._canvas_img_id = self.video_canvas.create_image(ox, oy, anchor=tk.NW, image=imgtk)
        else:
            self.video_canvas.coords(self._canvas_img_id, ox, oy)
            self.video_canvas.itemconfig(self._canvas_img_id, image=imgtk)

    def _start_live_analysis(self):
        if not os.path.exists(self.video_path):
            messagebox.showerror("Hata", "Video dosyası bulunamadı!")
            return
        if self.cap:
            self.cap.release()

        play_source = self.video_path
        if os.path.basename(self.video_path) == "video.mp4" and os.path.exists("video_720p.mp4"):
            play_source = "video_720p.mp4"

        self.cap = cv2.VideoCapture(play_source)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 690

        cache_file = "video_detections.json" if os.path.basename(self.video_path) == "video.mp4" else ""
        self.analyzer = BarPathAnalyzer(
            model_path=self.model_path,
            detections_cache_path=cache_file,
            track_mode=self.track_mode_var.get(),
            trail_length=int(self.trail_scale.get()),
            line_thickness=int(self.thick_scale.get()),
            vel_threshold=float(self.vel_scale.get()),
        )
        self.is_playing = True
        self.frame_idx = 0
        self.btn_pause.configure(text="⏸ Duraklat", bg="#16a34a")
        self.status_var.set("🟢 Canlı analiz çalışıyor! Sol panelden tüm ayarları anlık değiştirebilirsiniz.")
        self._loop_frame()

    def _toggle_pause(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_pause.configure(text="⏸ Duraklat", bg="#16a34a")
            self.status_var.set("▶️ Oynatılıyor...")
            self._loop_frame()
        else:
            self.btn_pause.configure(text="▶️ Devam Et", bg="#eab308")
            self.status_var.set("⏸ Duraklatıldı. Ayarları değiştirebilir veya videoda plakaya tıklayabilirsiniz.")

    def _loop_frame(self):
        if not self.is_playing or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            # Video bittiğinde otomatik başa sar (veya duraklat)
            self.is_playing = False
            self.btn_pause.configure(text="▶️ Devam Et", bg="#eab308")
            self.status_var.set("✅ Set analizi tamamlandı! 'Baştan Başlat' ile tekrar izleyebilirsiniz.")
            return

        self.last_raw_frame = frame
        annotated, _ = self.analyzer.process_frame(frame, self.frame_idx, self.fps)
        self.frame_idx += 1
        self.frame_info_var.set(f"Kare: {self.frame_idx} / {self.total_frames}  |  Tekrar: {self.analyzer.rep_count}")

        self._display_cv_frame(annotated)
        self.root.after(28, self._loop_frame)


if __name__ == "__main__":
    root = tk.Tk()
    app = BarPathStudioApp(root)
    root.mainloop()
