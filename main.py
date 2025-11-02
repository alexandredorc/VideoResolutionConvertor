import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from queue import Queue
import time
import math

# =========================
# Résolutions disponibles
# =========================
resolutions = {
    "144p": (256, 144),
    "240p": (426, 240),
    "360p": (640, 360),
    "480p": (854, 480),
    "720p (HD)": (1280, 720),
    "1080p (Full HD)": (1920, 1080),
    "1440p (2K/QHD)": (2560, 1440),
    "2160p (4K/UHD)": (3840, 2160),
    "4320p (8K/UHD)": (7680, 4320)
}

# =========================
# Chemins vers les exécutables FFmpeg
# =========================
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS  # type: ignore
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

FFMPEG_BIN = os.path.join(base_path, "ffmpeg.exe")
FFPROBE_BIN = os.path.join(base_path, "ffprobe.exe")

# =========================
# Configuration
# =========================
cpu_threads = os.cpu_count() or 4
VIDEO_EXTS = ('.mp4', '.mkv', '.avi', '.mov', '.m4v')

class VideoConverterApp:
    def __init__(self, root):
        # File de logs
        self.log_queue = Queue()
        self.root = root
        self.cancel_flag = threading.Event()
        self._current_proc = None
        self._encoders_cache = None
        self.selected_files = []  # liste de fichiers choisis via le bouton "Fichier"


        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Fichier de log
        try:
            self.log_file = open(os.path.join(base_path, 'convert.log'), 'a', encoding='utf-8')
        except Exception:
            self.log_file = None

        # Task pour vider la queue de logs vers la TextBox
        self.root.after(100, self._drain_log_queue)

        self.check_system_ready()

    # =========================
    # UI
    # =========================
    def setup_ui(self):
        self.root.title("Convertisseur Vidéo Avancé - GPU/CPU Auto")
        self.root.geometry("980x680")
        self.root.resizable(False, False)

        frame_top = tk.Frame(self.root)
        frame_top.pack(pady=10, fill=tk.X)

        self.entry_path = tk.Entry(frame_top, width=60)
        self.entry_path.pack(side=tk.LEFT, padx=5)

        self.btn_file = tk.Button(frame_top, text="📄 Fichier", command=self.select_file)
        self.btn_file.pack(side=tk.LEFT, padx=2)

        self.btn_folder = tk.Button(frame_top, text="📁 Dossier", command=self.select_folder)
        self.btn_folder.pack(side=tk.LEFT, padx=2)

        self.btn_check = tk.Button(frame_top, text="🛠️ Vérifier", command=self.check_system)
        self.btn_check.pack(side=tk.LEFT, padx=2)

        tk.Label(frame_top, text="Résolution :").pack(side=tk.LEFT, padx=5)
        self.combo_res = ttk.Combobox(
            frame_top,
            values=list(resolutions.keys()),
            state="readonly",
            width=18
        )
        self.combo_res.current(list(resolutions.keys()).index("1080p (Full HD)"))
        self.combo_res.pack(side=tk.LEFT, padx=2)

        self.btn_start = tk.Button(self.root, text="▶️ Démarrer", command=self.start_conversion, bg="#2ecc71", fg="white")
        self.btn_start.pack(pady=5)

        self.btn_cancel = tk.Button(self.root, text="⛔ Annuler", command=self.cancel, bg="#e74c3c", fg="white", state="disabled")
        self.btn_cancel.pack(pady=2)

        # Progression
        frame_prog = tk.Frame(self.root)
        frame_prog.pack(pady=4, fill=tk.X, padx=10)
        tk.Label(frame_prog, text="Progression fichier:").pack(anchor="w")
        self.pb_file = ttk.Progressbar(frame_prog, orient="horizontal", mode="determinate", length=940, maximum=100)
        self.pb_file.pack(pady=2)
        self.lbl_file = tk.Label(frame_prog, text="0%")
        self.lbl_file.pack(anchor="e")

        tk.Label(frame_prog, text="Progression globale:").pack(anchor="w")
        self.pb_total = ttk.Progressbar(frame_prog, orient="horizontal", mode="determinate", length=940, maximum=100)
        self.pb_total.pack(pady=2)
        self.lbl_total = tk.Label(frame_prog, text="0%")
        self.lbl_total.pack(anchor="e")

        # Sortie logs
        self.text_output = scrolledtext.ScrolledText(self.root, width=120, height=22, state="normal")
        self.text_output.pack(padx=10, pady=6)

    def _drain_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.text_output.insert(tk.END, msg + "\n")
                self.text_output.see(tk.END)
        except Exception:
            pass
        self.root.after(100, self._drain_log_queue)

    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        self.log_queue.put(line)
        if self.log_file:
            try:
                self.log_file.write(line + "\n")
                self.log_file.flush()
            except Exception:
                pass

    def update_buttons_state(self, running: bool):
        if running:
            self.btn_start.config(state="disabled")
            self.btn_cancel.config(state="normal")
            self.btn_check.config(state="disabled")
            self.btn_file.config(state="disabled")
            self.btn_folder.config(state="disabled")
        else:
            self.btn_start.config(state="normal")
            self.btn_cancel.config(state="disabled")
            self.btn_check.config(state="normal")
            self.btn_file.config(state="normal")
            self.btn_folder.config(state="normal")

    def cancel(self):
        self.cancel_flag.set()
        self.log("⛔ Annulation demandée…")
        proc = self._current_proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    # =========================
    # Sélection fichiers
    # =========================
    def select_file(self):
        paths = filedialog.askopenfilenames(
            title="Sélectionner des fichiers vidéo",
            filetypes=[
                ("Vidéos", "*.mp4 *.mkv *.avi *.mov *.m4v"),
                ("Tous les fichiers", "*.*"),
            ]
        )
        if not paths:
            return

        self.selected_files = list(paths)

        # Affichage dans l'entry
        if len(paths) == 1:
            display = paths[0]
        else:
            first = os.path.basename(paths[0])
            display = f"{first} + {len(paths) - 1} autres"

        self.entry_path.delete(0, tk.END)
        self.entry_path.insert(0, display)

        # Petit log
        self.log(f"📄 {len(paths)} fichier(s) sélectionné(s).")


    def select_folder(self):
        folder = filedialog.askdirectory(title="Sélectionner un dossier")
        if not folder:
            return
        self.selected_files = []  # on annule la sélection multiple
        self.entry_path.delete(0, tk.END)
        self.entry_path.insert(0, folder)
        self.log(f"📁 Dossier sélectionné: {folder}")

    # =========================
    # Vérifications système
    # =========================
    def check_system_ready(self):
        # Vérifie la présence de FFmpeg/FFprobe
        for exe, name in [(FFMPEG_BIN, "FFmpeg"), (FFPROBE_BIN, "FFprobe")]:
            if not os.path.exists(exe):
                self.log(f"❌ {name} introuvable : {exe}")
            else:
                self.log(f"✅ {name} détecté : {exe}")

    def check_system(self):
        self.log("🔍 Vérification du système…")
        # FFmpeg version
        try:
            r = subprocess.run([FFMPEG_BIN, '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if r.returncode == 0:
                first = r.stdout.splitlines()[0] if r.stdout else "FFmpeg"
                self.log(f"✅ {first}")
            else:
                self.log("❌ FFmpeg ne répond pas correctement")
        except Exception as e:
            self.log(f"❌ Erreur FFmpeg : {e}")

        # Encoders
        encs = self.get_encoders()
        # GPU support
        hwaccels = self.get_hwaccels()
        self.log(f"ℹ️ Méthodes HW: {', '.join(hwaccels) if hwaccels else 'aucune'}")
        gpu_encs = [e for e in ('hevc_amf', 'h264_amf') if e in encs]
        if gpu_encs:
            self.log(f"✅ Encodeurs GPU AMF dispo: {', '.join(gpu_encs)}")
        else:
            self.log("⚠️ Aucun encodeur AMF détecté")

        # CPU libs
        for lib in ('libx265', 'libx264'):
            self.log(f"{'✅' if lib in encs else '⚠️'} {lib} {'disponible' if lib in encs else 'absent'}")

    def get_hwaccels(self):
        try:
            r = subprocess.run([FFMPEG_BIN, '-hwaccels'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            out = []
            for line in r.stdout.splitlines():
                line = line.strip()
                if line and not line.lower().startswith("hardware acceleration"):
                    out.append(line)
            return out
        except Exception:
            return []
        
    def make_res_suffix(width: int, height: int) -> str:
        """Retourne un suffixe de type '_1920x1080' pour le nom de fichier."""
        return f"{int(width)}x{int(height)}"


    def get_encoders(self):
        if self._encoders_cache is not None:
            return self._encoders_cache
        try:
            r = subprocess.run([FFMPEG_BIN, '-hide_banner', '-encoders'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            encs = set()
            for line in r.stdout.splitlines():
                if " V" in line[:3] or "EV" in line[:3] or "V." in line[:3]:
                    parts = line.split()
                    if parts:
                        encs.add(parts[1])
            self._encoders_cache = encs
            return encs
        except Exception:
            return set()

    def has_encoder(self, name):
        return name in self.get_encoders()

    # =========================
    # Détection/Tests GPU
    # =========================
    def can_encode_amf(self, encoder: str, width: int, height: int, fps: int = 30):
        # Essai 1 frame synthétique pour vérifier que l'encodeur AMF accepte la résolution
        test_cmd = [
            FFMPEG_BIN, '-y',
            '-f', 'lavfi', '-i', f"testsrc2=size={width}x{height}:rate={fps}",
            '-frames:v', '1',
            '-c:v', encoder,
            '-usage', 'transcoding',
            '-quality', 'balanced',
            '-profile', 'main',
            '-tier', 'main',
            '-level', '5.1' if fps <= 30 else '5.2',
            '-pix_fmt', 'yuv420p',
            '-f', 'null', '-'
        ]
        try:
            r = subprocess.run(test_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return r.returncode == 0
        except Exception:
            return False

    # =========================
    # Conversion
    # =========================
    def start_conversion(self):
        # Récupère ce que l'utilisateur a mis/choisi
        path = self.entry_path.get().strip()

        # Construit la liste des fichiers à traiter
        files = []

        # 1) Priorité aux fichiers sélectionnés via le bouton "Fichier"
        if self.selected_files:
            files = [p for p in self.selected_files if p.lower().endswith(VIDEO_EXTS)]

        # 2) Sinon, si un dossier est indiqué, on scanne
        elif path and os.path.isdir(path):
            for root_dir, _, fnames in os.walk(path):
                for fn in fnames:
                    if fn.lower().endswith(VIDEO_EXTS):
                        files.append(os.path.join(root_dir, fn))

        # 3) Sinon, si un seul fichier est indiqué
        elif path and os.path.isfile(path):
            if path.lower().endswith(VIDEO_EXTS):
                files = [path]
            else:
                messagebox.showwarning("Attention", "Le fichier sélectionné n'est pas une vidéo prise en charge.")
                return

        else:
            messagebox.showwarning("Attention", "Veuillez sélectionner un fichier, plusieurs fichiers, ou un dossier.")
            return

        if not files:
            messagebox.showwarning("Attention", "Aucun fichier vidéo pris en charge trouvé.")
            return

        # Optionnel: tri et déduplication
        files = sorted(list(dict.fromkeys(files)))

        # Reset UI/progress
        self.cancel_flag.clear()
        self.pb_file['value'] = 0
        self.pb_total['value'] = 0
        self.lbl_file.config(text="0%")
        self.lbl_total.config(text="0%")
        self.text_output.delete(1.0, tk.END)

        self.log(f"🔍 Début de la conversion ({len(files)} fichier(s))…")
        self.update_buttons_state(True)

        # Thread de traitement
        thread = threading.Thread(target=self.process_files, args=(files,), daemon=True)
        thread.start()


    def process_files(self, files):
        total_files = len(files)
        for idx, fpath in enumerate(files, 1):
            if self.cancel_flag.is_set():
                self.log("⛔ Conversion annulée par l’utilisateur.")
                break

            self.log("=" * 60)
            self.log(f"========== Fichier {idx}/{total_files} ==========")
            ok = self.convert_one(fpath)
            if not ok:
                self.log(f"❌ Échec de la conversion : {os.path.basename(fpath)}")
            # progression globale
            self.update_total_progress(idx, total_files)

        self.root.after(0, lambda: self.update_buttons_state(False))
        if not self.cancel_flag.is_set():
            self.log("✅ Toutes les conversions sont terminées !")
        else:
            self.log("ℹ️ Conversion interrompue.")

    def update_total_progress(self, done, total):
        pct = int(done / total * 100)
        self.pb_total['value'] = pct
        self.lbl_total.config(text=f"{pct}%")

    def get_video_duration_seconds(self, input_file):
        try:
            r = subprocess.run(
                [FFPROBE_BIN, '-v', 'error', '-select_streams', 'v:0',
                 '-show_entries', 'format=duration', '-of', 'default=nw=1:nk=1', input_file],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15
            )
            val = r.stdout.strip()
            return float(val) if val else 0.0
        except Exception:
            return 0.0

    def build_vf(self, target_width, target_height):
        # Conserver l'aspect ratio + padding noir si nécessaire
        vf = (
            f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
            f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:color=black"
        )
        return vf

    def next_output_path(self, input_file, suffix):
        out_dir = os.path.join(os.path.dirname(input_file), "converted")
        os.makedirs(out_dir, exist_ok=True)
        base, _ = os.path.splitext(os.path.basename(input_file))
        output = os.path.join(out_dir, f"{base}_{suffix}.mp4")
        i = 1
        while os.path.exists(output):
            output = os.path.join(out_dir, f"{base}_{suffix}({i}).mp4")
            i += 1
        return output

    def convert_one(self, input_file):
        # Résolution cible
        res_key = self.combo_res.get()
        target_w, target_h = resolutions.get(res_key, (1920, 1080))
        vf = self.build_vf(target_w, target_h)

        # Durée pour la progression
        total_seconds = self.get_video_duration_seconds(input_file)

        # Détection encodeurs
        encs = self.get_encoders()

        # Choix GPU AMF si dispo
        candidates = []
        amf_tested = set()

        if 'hevc_amf' in encs and self.can_encode_amf('hevc_amf', target_w, target_h):
            candidates.append(('hevc_amf', 'gpu'))
            amf_tested.add('hevc_amf')
        if 'h264_amf' in encs and self.can_encode_amf('h264_amf', target_w, target_h):
            candidates.append(('h264_amf', 'gpu'))
            amf_tested.add('h264_amf')

        # CPU fallbacks
        if 'libx265' in encs:
            candidates.append(('libx265', 'cpu'))
        if 'libx264' in encs:
            candidates.append(('libx264', 'cpu'))

        if not candidates:
            self.log("❌ Aucun encodeur compatible détecté (hevc_amf/h264_amf/libx265/libx264 manquants).")
            return False

        self.log(f"🎯 Candidats: {', '.join([c[0] for c in candidates])}")

        # Boucle d'essai par priorité
        for enc, kind in candidates:
            if self.cancel_flag.is_set():
                return False

            suffix = f"{target_w}x{target_h}"
            output_file = self.next_output_path(input_file, suffix)
            self.log(f"[{kind.upper()}] Conversion {os.path.basename(input_file)} → {enc}…")

            cmd = self.build_cmd(input_file, output_file, enc, kind, vf, target_w, target_h)
            ok = self.run_ffmpeg(cmd, total_seconds)

            if ok:
                self.log(f"✅ Terminé ({enc}) : {output_file}")
                return True
            else:
                self.log(f"⚠️ Échec avec {enc}, essai du suivant…")

        return False

    def build_cmd(self, input_file, output_file, encoder, kind, vf, target_w, target_h):
        # Audio: AAC 192k par défaut
        audio_args = ['-c:a', 'aac', '-b:a', '192k']

        if kind == 'gpu' and encoder in ('hevc_amf', 'h264_amf'):
            # Paramètres AMF stables pour 2K/4K
            amf_common = [
                '-usage', 'transcoding',
                '-quality', 'balanced',
                '-profile', 'main',
                '-tier', 'main',
                '-level', '5.1',  # mets 5.2 si 60 fps garanti
                '-pix_fmt', 'yuv420p',
                '-rc', 'vbr_latency',
                '-b:v', '0',
                '-qp_i', '23',
                '-qp_p', '23'
            ]
            # NOTE: pas de -hwaccel forcé pour éviter des gels d’initialisation
            cmd = [
                FFMPEG_BIN, '-y',
                '-i', input_file,
                '-vf', vf,
                '-c:v', encoder, *amf_common,
                *audio_args,
                output_file
            ]
            return cmd

        # CPU encoders
        if encoder == 'libx265':
            v_args = ['-c:v', 'libx265', '-preset', 'medium', '-crf', '22', '-pix_fmt', 'yuv420p', '-threads', str(cpu_threads)]
        elif encoder == 'libx264':
            v_args = ['-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', '-threads', str(cpu_threads)]
        else:
            v_args = ['-c:v', encoder]

        cmd = [
            FFMPEG_BIN, '-y',
            '-i', input_file,
            '-vf', vf,
            *v_args,
            *audio_args,
            output_file
        ]
        return cmd


    def run_ffmpeg(self, cmd, total_seconds):
        # Ajoute -progress pipe:1 + loglevel réduit
        full_cmd = cmd[:]
        insert_at = 1
        full_cmd[insert_at:insert_at] = ['-progress', 'pipe:1', '-nostats', '-hide_banner', '-loglevel', 'error']

        self.pb_file['value'] = 0
        self.lbl_file.config(text="0%")

        import collections
        from queue import Queue, Empty

        stdout_q = Queue()
        stderr_tail = collections.deque(maxlen=80)

        def _reader_stdout(p, q):
            try:
                for line in iter(p.stdout.readline, ''):
                    if not line:
                        break
                    q.put(line)
            except Exception:
                pass

        def _reader_stderr(p, tail):
            try:
                for line in iter(p.stderr.readline, ''):
                    if not line:
                        break
                    s = line.rstrip()
                    if s:
                        tail.append(s)
            except Exception:
                pass

        try:
            proc = subprocess.Popen(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
            self._current_proc = proc

            t_out = threading.Thread(target=_reader_stdout, args=(proc, stdout_q), daemon=True)
            t_err = threading.Thread(target=_reader_stderr, args=(proc, stderr_tail), daemon=True)
            t_out.start(); t_err.start()

            start_time = time.time()
            last_progress = None       # dernière fois où on a vu out_time_ms
            startup_timeout = 20.0     # pas de progrès du tout → on tue
            stall_timeout = 30.0       # plus de progrès en cours → on tue

            cur_sec = 0.0
            last_ui = 0.0

            while True:
                if self.cancel_flag.is_set():
                    try: proc.terminate()
                    except Exception: pass
                    self._current_proc = None
                    return False

                # essaie de lire une ligne de progress avec timeout court
                try:
                    line = stdout_q.get(timeout=0.5)
                except Empty:
                    line = None

                now = time.time()

                if line:
                    line = line.strip()
                    if line.startswith('out_time_ms='):
                        try:
                            ms = int(line.split('=')[1])
                            cur_sec = ms / 1_000_000.0
                            last_progress = now
                            if total_seconds > 0:
                                pct = max(0, min(100, int(cur_sec / total_seconds * 100)))
                            else:
                                pct = min(100, int(self.pb_file['value']) + 1)

                            if now - last_ui > 0.1:
                                self.pb_file['value'] = pct
                                self.lbl_file.config(text=f"{pct}%")
                                last_ui = now
                        except Exception:
                            pass
                    # d'autres clés de -progress ne nous intéressent pas ici

                # Watchdog démarrage
                if last_progress is None:
                    if now - start_time > startup_timeout:
                        self.log(f"⏱️ Aucun progrès après {int(startup_timeout)}s → arrêt et fallback.")
                        try: proc.terminate()
                        except Exception: pass
                        proc.wait(timeout=5)
                        self.log("---- FFmpeg (dernières lignes) ----")
                        for l in list(stderr_tail)[-30:]:
                            self.log(l)
                        self.log("---- fin FFmpeg ----")
                        self._current_proc = None
                        return False
                else:
                    # Watchdog “stall”
                    if now - last_progress > stall_timeout:
                        self.log(f"⏱️ Bloqué depuis {int(stall_timeout)}s (plus de progrès) → arrêt et fallback.")
                        try: proc.terminate()
                        except Exception: pass
                        proc.wait(timeout=5)
                        self.log("---- FFmpeg (dernières lignes) ----")
                        for l in list(stderr_tail)[-30:]:
                            self.log(l)
                        self.log("---- fin FFmpeg ----")
                        self._current_proc = None
                        return False

                # Sortie propre si le process est terminé
                if proc.poll() is not None:
                    rc = proc.returncode
                    self._current_proc = None
                    if rc != 0:
                        self.log("---- FFmpeg (dernières lignes) ----")
                        for l in list(stderr_tail)[-30:]:
                            self.log(l)
                        self.log("---- fin FFmpeg ----")
                        return False
                    else:
                        self.pb_file['value'] = 100
                        self.lbl_file.config(text="100%")
                        return True

        except Exception as e:
            self._current_proc = None
            self.log(f"Erreur FFmpeg : {e}")
            return False


    # =========================
    # Fermeture
    # =========================
    def on_closing(self):
        self.cancel_flag.set()
        if self._current_proc and self._current_proc.poll() is None:
            try:
                self._current_proc.terminate()
            except Exception:
                pass
        try:
            if self.log_file:
                self.log_file.close()
        except Exception:
            pass
        self.root.destroy()


# =========================
# Lancement
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = VideoConverterApp(root)
    root.mainloop()
