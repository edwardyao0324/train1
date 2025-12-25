import traceback
import torch
import os
from flask import jsonify
import time
from transformers import TextIteratorStreamer
import threading
from typing import List, Optional
import cv2
from PIL import Image, ImageOps
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoTokenizer
import pyttsx3
import opencc
import pygame
import mediapipe as mp
import math
import re
from flask import Flask, Response, render_template_string, request
import uuid
import queue

os.makedirs("tts_files", exist_ok=True)
#==================== TTS 安全播放（blocking）============================
tts_queue = queue.Queue()
tts_engine_lock = threading.Lock()
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 110) # 語速 愈小愈慢
tts_engine.setProperty('volume', 1.0) #音量最大了

# ---------------- TTS queue ----------------
def tts_worker():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        try:
            with tts_engine_lock:
                tts_engine.say(text)
                tts_engine.runAndWait()  # blocking
        except Exception as e:
            print("[tts_worker error]", e)
        tts_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

def tts_play_nonblocking(text: str):
    if text.strip():
        tts_queue.put(text)
        
def _tts_play(text):
    try:
        with tts_engine_lock:
            tts_engine.say(text)
            tts_engine.runAndWait()
    except Exception as e:
        print("[tts_play error]", e)

#====================關掉一警告============================
os.environ['GLOG_minloglevel'] = '2'
os.environ['WEBRTC_DISABLE_GRAPHITE'] = '1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# -------------------- 基本鎖與 TTS queue --------------------
model_lock = threading.Lock()
MODEL_READY = False  # 全域 flag，模型是否已載入
model_loading = False

# -------------------- 下載後路徑會變 要自己改--------------------
COUNTDOWN_AUDIO_PATH = r"C:\Users\Edward\Desktop\Deep\train1\tts_files\倒數.wav" #下載後路徑會變 要自己改
SHUTTER_AUDIO_PATH = r"C:\Users\Edward\Desktop\Deep\train1\tts_files\快門聲.MP3" #下載後路徑會變 要自己改
STARTUP_AUDIO_PATH = r"C:\Users\Edward\Desktop\Deep\train1\tts_files\讚.wav" #下載後路徑會變 要自己改


# pygame 初始化（用來播放 mp3）
try:
    if not pygame.mixer.get_init():
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=4096)
except Exception as e:
    print("[pygame init error]", e)

# ----------------Mediapipe 初始化---------------------
mp_drawing = mp.solutions.drawing_utils  # 繪製手部關鍵點
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands  # mediapipe 手部偵測模組
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------- CONFIG ----------------
MODEL_PATH = "D:/DATASET" #下載後路徑會變要自己改
PORT = 7860
HAS_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if HAS_CUDA else "cpu"
AUTO_CAPTURE_INTERVAL = 3.0  # 自動拍照間隔（秒）

os.makedirs("captures", exist_ok=True)

# ---------------- GLOBALS ----------------
chat_history: List = []  # each entry: [label, description]
chat_lock = threading.Lock()
is_generating = False # 是否正在生成描述
# 防止連續 GOOD 觸發
good_cooldown = False
GOOD_COOLDOWN_SEC = 5.0  # >= 倒數 + 快門秒數 才截圖
hand_good_detected = False
last_good_time = 0.0
latest_desc = ""
latest_desc_lock = threading.Lock()

# ---------------- model/process/tokenizer 初始 ----------------
model = None
converter = opencc.OpenCC("s2t.json")
last_auto_capture = 0.0
frame_shared = None
frame_lock = threading.Lock()
is_counting_down = False
auto_audio_path: Optional[str] = None
manual_audio_path: Optional[str] = None
audio_path_lock = threading.Lock()
processor = None
tokenizer = None
frame_clean = None               # 把截圖good去除
frame_clean_lock = threading.Lock()

cap = cv2.VideoCapture(0)

# ---------------- UTIL ----------------
def to_traditional(s: str) -> str:
    try:
        return converter.convert(s)
    except:
        return s


def normalize_image(img: Image.Image, max_side: int = 1280) -> Image.Image:
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    return img

# ---------------- Lazy load ----------------
def lazy_load_qwen_bg():
    global model, processor, tokenizer, MODEL_READY, model_loading

    with model_lock:
        if model is not None or model_loading:
            return
        model_loading = True

    success = False
    try:
        print("[init] 開始載入本地DATASET...")

        if not torch.cuda.is_available():
            device = torch.device("cpu")
            dtype = torch.float32
            print("[init] CUDA 不可用，使用 CPU")
        else:
            device = torch.device("cuda:0")
            dtype = torch.float16  # 用 float16 節省 GPU 記憶體
            print(f"[init] CUDA 可用，使用 GPU: {torch.cuda.get_device_name(0)}")

        # 強制全部放到 GPU 0
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=dtype,
            device_map={"": 0},  # 全部放 GPU0
            trust_remote_code=True
        )

        # debug: 檢查第一個參數的 device
        first_param = next(model.parameters())
        print("[init] 模型第一個參數 device =", first_param.device)

        # processor
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

        # tokenizer
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

        success = True
        print("[init] DATASET 載入完成")

    except Exception as e:
        print("[Fatal] DATASET 模型載入失敗")
        print(e)
        traceback.print_exc()
        model = None
        processor = None
        tokenizer = None
    finally:
        MODEL_READY = model is not None and processor is not None and tokenizer is not None
        model_loading = False
        print("[MODEL_READY]", MODEL_READY)

        # debug 列出前 5 個參數
        if model:
            for i, p in enumerate(list(model.parameters())[:5]):
                print(f"[DEBUG] param {i} device = {p.device}")

# ================== 缺失定義與輔助函式 ==================
def update_latest_desc_from_chat():
    global latest_desc
    with chat_lock, latest_desc_lock:
        if chat_history:
            latest_desc = chat_history[-1][1]

def cut_to_user_question(user_text, model_output, has_image=False):

    return model_output

def normalize_china(s: str) -> str:

    try:
        return converter.convert(s)
    except Exception:
        return s

# ---------------- sound play helper ----------------
def play_sound_nonblocking(path):
    try:
        sound = pygame.mixer.Sound(path)
        sound.play()
    except Exception as e:
        print("[play_sound_nonblocking error]", e)

def play_countdown_sound():
    if os.path.exists(COUNTDOWN_AUDIO_PATH):
        play_sound_nonblocking(COUNTDOWN_AUDIO_PATH)

def play_shutter_sound():
    if os.path.exists(SHUTTER_AUDIO_PATH):
        play_sound_nonblocking(SHUTTER_AUDIO_PATH)

def play_startup_sound():
    if os.path.exists(STARTUP_AUDIO_PATH):
        play_sound_nonblocking(STARTUP_AUDIO_PATH)

# ---------------- save frame helper ----------------
def save_frame_and_get_pil(frame_np) -> Image.Image:
    """
    將 OpenCV BGR frame 轉成 PIL Image，並做 normalize
    """
    try:
        # OpenCV BGR -> RGB
        rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil = normalize_image(pil)

        # 自動儲存到 captures
        fname = f"captures/{uuid.uuid4().hex}.jpg"
        pil.save(fname, format="JPEG", quality=92)

        return pil
    except Exception as e:
        print("[save_frame_and_get_pil] error:", e)
        return None

# -------------------- Helpers & Prompts --------------------
SYSTEM_PROMPT = (
    "任何簡體字都必須轉為繁體後再輸出。"
    "針對圖片內容場景、人物動作、穿著、打扮進行描述、分析"
    "若圖片出現人物可以進行身分說明還有年齡說明"
    "必須整合成一篇流暢、連貫的描述，"
)

def detect_language(text):
    if not text:
        return "zh"
    a = sum(1 for c in text if ord(c) < 128)
    return "en" if a / max(len(text), 1) > 0.6 else "zh"

# ----------------------------------------------------
#                推論核心（整合版）
# ----------------------------------------------------

def _build_messages(user_text: str, images: List[Image.Image], history: List[List[str]]):
    if not MODEL_READY:
        raise RuntimeError("MODEL_NOT_READY")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        for turn in history:
            if not isinstance(turn, (list, tuple)):
                continue
            if len(turn) > 0:
                messages.append({"role": "user", "content": turn[0]})
            if len(turn) > 1:
                messages.append({"role": "assistant", "content": turn[1]})

    content = []
    for img in images or []:
        content.append({"type": "image", "image": normalize_image(img)})
    content.append({"type": "text", "text": user_text or ""})

    messages.append({"role": "user", "content": content})
    return messages

# ---------------- processor prepare inputs ----------------
def _processor_prepare_inputs(messages):
    imgs = []
    try:
        last = messages[-1]["content"]
        for c in last:
            if isinstance(c, dict) and c.get("type") == "image":
                img_obj = c.get("image")
                if img_obj is not None:
                    try:
                        img_obj = normalize_image(img_obj)
                        imgs.append(img_obj)
                    except Exception as e:
                        print("[processor] normalize_image fail:", e)
    except Exception as e:
        print("[processor] extract images fail:", e)
        imgs = []

    prompt_text = ""
    try:
        prompt_text = processor.apply_chat_template(messages, tokenize=False)
    except Exception as e:
        print("[processor] apply_chat_template fail:", e)
        prompt_text = str(messages)

    try:
        if imgs:
            inputs = processor(text=prompt_text, images=imgs, return_tensors="pt")
        else:
            inputs = processor(text=prompt_text, return_tensors="pt")
    except Exception as e:
        print("[processor] processor call fail:", e)
        inputs = {"input_ids": tokenizer(prompt_text, return_tensors="pt").input_ids}

    # ------------------ 強制搬到 model.device ------------------
    device = next(model.parameters()).device

    def move_to_device(obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device, non_blocking=True)
        elif isinstance(obj, dict):
            return {k: move_to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(move_to_device(v, device) for v in obj)
        else:
            return obj

    inputs = move_to_device(inputs, device)
    return inputs

def safe_model_generate(model, *args, **kwargs):
    print("[DEBUG] safe_model_generate called")
    print("[DEBUG] model device =", next(model.parameters()).device)

    for k in ("temperature", "top_p", "top_k"):
        if k in kwargs:
            kwargs.pop(k)

    def _move_to_device(obj, device):
        if isinstance(obj, torch.Tensor):
            try:
                return obj.to(device)
            except Exception:
                return obj
        elif isinstance(obj, dict):
            return {k: _move_to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            moved = [_move_to_device(v, device) for v in obj]
            return type(obj)(moved)
        else:
            return obj

    try:
        model_device = next(model.parameters()).device
    except Exception:
        model_device = getattr(model, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    new_args = tuple(_move_to_device(a, model_device) for a in args)
    new_kwargs = {k: _move_to_device(v, model_device) for k, v in kwargs.items()}

    try:
        return model.generate(*new_args, **new_kwargs)
    except RuntimeError as e:
        msg = str(e)
        if ('Expected all tensors to be on the same device' in msg) or ('Image features and image tokens do not match' in msg):
            try:
                try:
                    model_device = next(model.parameters()).device
                except Exception:
                    model_device = getattr(model, 'device', model_device)
                new_args = tuple(_move_to_device(a, model_device) for a in new_args)
                new_kwargs = {k: _move_to_device(v, model_device) for k, v in new_kwargs.items()}
                return model.generate(*new_args, **new_kwargs)
            except Exception:
                pass
        raise

def run_model(user_text, images, history):
    if not MODEL_READY:
        raise RuntimeError("模型尚未載入完成")

    messages = _build_messages(user_text, images, history)
    inputs = _processor_prepare_inputs(messages)

    output = safe_model_generate(
        model,
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        use_cache=True
    )

    seq = output[0] if isinstance(output, (list, tuple, torch.Tensor)) else output
    text = tokenizer.decode(seq, skip_special_tokens=True)

    # 僅保留 assistant 回應
    for key in ("Assistant:", "assistant:", "ASSISTANT:"):
        if key in text:
            text = text.split(key, 1)[-1]

    return to_traditional(text.strip())

def stream_model(user_text: str, images: List[Image.Image], history):
    if not MODEL_READY:
        yield "模型載入中，請稍候..."
        return

    msgs = _build_messages(user_text, images, history)
    inputs = _processor_prepare_inputs(msgs)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_special_tokens=True,
        skip_prompt=True
    )

    def gen():
        safe_model_generate(
            model,
            **inputs,
            max_new_tokens=1024,   # 可改大一點
            do_sample=False,
            streamer=streamer,
            use_cache=True
        )

    threading.Thread(target=gen, daemon=True).start()

    for chunk in streamer:
        if chunk:
            yield chunk   # 直接每個 chunk yield 回應

# ---------------- 手勢觸發拍照 queue ----------------
hand_capture_queue = queue.Queue()

def auto_capture_on_hand():
    if hand_capture_queue.qsize() > 0:
        return
    hand_capture_queue.put(True)

# ---------------- worker thread ----------------
def capture_worker():
    while True:
        _ = hand_capture_queue.get()   # 只表示「拍照請求」
        countdown_and_capture()
        hand_capture_queue.task_done()

threading.Thread(target=capture_worker, daemon=True).start() # 啟動 worker thread

# ---------------- 倒數 + 拍照 + TTS 描述 ----------------
def countdown_and_capture():
    def capture_sequence():
        print("[DEBUG] countdown_and_capture start")
        # 倒數音
        if os.path.exists(COUNTDOWN_AUDIO_PATH):
            sound = pygame.mixer.Sound(COUNTDOWN_AUDIO_PATH)
            sound.play()
            while pygame.mixer.get_busy():
                time.sleep(0.05)

        # 快門音
        if os.path.exists(SHUTTER_AUDIO_PATH):
            sound = pygame.mixer.Sound(SHUTTER_AUDIO_PATH)
            sound.play()
            while pygame.mixer.get_busy():
                time.sleep(0.05)

        # 抓 frame
        frame_to_save = None
        for _ in range(20):   # 最多 2 秒
            with frame_clean_lock:
                print("[DEBUG] frame_clean =", frame_clean is not None)  # DEBUG
                if frame_clean is not None:
                    frame_to_save = frame_clean.copy()
                    break
            time.sleep(0.05)

        if frame_to_save is None:
            print("[CAPTURE] timeout: no frame_clean")
            with chat_lock:
                chat_history.append(["(拍照)", "無可用畫面"])
            return

        pil_image = save_frame_and_get_pil(frame_to_save)
        if pil_image:
            print("[DEBUG] start_image_description from countdown")
            start_image_description(pil_image, source="(拍照)")

    threading.Thread(target=capture_sequence, daemon=True).start()

def start_image_description(pil_image: Image.Image, source="(拍照)"):
    global is_generating
    with chat_lock:
        chat_history.append([source, ""])

    if not MODEL_READY:
        with chat_lock:
            chat_history[-1][1] = "模型尚未準備好"
        return

    is_generating = True

    def strip_markdown(text: str) -> str:
        # 去掉 Markdown，合併換行
        text = text.replace("\n", " ")
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'`(.*?)`', r'\1', text)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        return text.strip()
    
    def worker():
        global is_generating
        text_buf = ""

        # 每個 chunk 都清理掉 Markdown，累積到文字框
        for chunk in stream_model("請你描述這張圖片的內容。", [pil_image], chat_history):
            if not chunk:
                continue

            clean_chunk = strip_markdown(chunk)
            text_buf += clean_chunk + " "  # 保持連續

            # 更新文字框
            with chat_lock:
                chat_history[-1][1] = text_buf
            update_latest_desc_from_chat()

    # ---------------- 播放生成音效（非阻塞） ----------------
    def play_generate_sound():
        gen_sound_path = r"C:\Users\Edward\Desktop\Deep\train1\tts_files\生成.wav" #下載後路徑會變 要自己改
        if os.path.exists(gen_sound_path):
            try:
                sound = pygame.mixer.Sound(gen_sound_path)
                sound.play()
            except Exception as e:
                print("[play_generate_sound error]", e)

    threading.Thread(target=play_generate_sound, daemon=True).start()

    # ---------------- 文字生成 worker ----------------
    def worker():
        global is_generating
        text_buf = ""

        # 每個 chunk 都清理掉 Markdown，累積到文字框
        for chunk in stream_model("請你描述這張圖片的內容。", [pil_image], chat_history):
            if not chunk:
                continue

            clean_chunk = strip_markdown(chunk)
            text_buf += clean_chunk + " "  # 保持連續

            # 更新文字框
            with chat_lock:
                chat_history[-1][1] = text_buf
            update_latest_desc_from_chat()

        # 生成結束後，把整段文字送到 TTS queue，一次播完
        tts_play_nonblocking(text_buf.strip())

        # 最終更新文字框
        with chat_lock:
            chat_history[-1][1] = text_buf.strip()
        is_generating = False

    threading.Thread(target=worker, daemon=True).start()

# ---------------- Flask App ----------------
app = Flask(__name__)  

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<title>盲人輔助視覺系統</title>
<meta charset="utf-8" />
<style>
html, body { height: 100%; margin:0; padding:0; }
body {
    background-color: #000;
    font-family: Arial, Helvetica, sans-serif;
    color: #fff;
    display: flex;
    justify-content: center;
}
#container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 20px;
    max-width: 1000px;
    margin: 20px auto;
    width: 95%;
}
#model_output {
    width: 100%;
    min-height: 300px;
    border: 1px solid #999;
    padding: 12px;
    font-size: 18px;
    background: #f9f9f9;
    color: #000;
    border-radius: 8px;
    white-space: pre-wrap;
    overflow-y: auto; 
}
#camera_upload_area {
    display: flex;
    gap: 15px;
    width: 100%;
    justify-content: center;
}
#camera_view img {
    max-width: 100%;
    border-radius: 8px;
}
#send_btn {
    font-size: 20px;
    padding: 10px 24px;
    border-radius: 8px;
    cursor: pointer;
    background-color: #444;
    color: #fff;
    border: none;
}
#send_btn:hover {
    background-color: #666;
}
</style>
</head>

<body>
<div id="container">

<h3>盲人輔助視覺系統</h3>

<div id="model_output">系統啟動完成</div>

<div id="camera_upload_area">
    <div id="camera_view">
        <img src="{{ url_for('video_feed') }}">
    </div>
    <div>
        <input type="file" id="img_file" accept="image/*">
    </div>
</div>

<button id="send_btn">送出</button>

<script>
window.addEventListener('DOMContentLoaded', () => {
    const output = document.getElementById("model_output");
    const sendBtn = document.getElementById("send_btn");
    const imgFileInput = document.getElementById("img_file");

    // 送出按鈕
    sendBtn.addEventListener("click", async () => {
        output.textContent = "生成中…";

        const file = imgFileInput.files[0];

        try {
            if (file) {
                const formData = new FormData();
                formData.append("file", file);
                await fetch("/upload_image", { method: "POST", body: formData });
                imgFileInput.value = "";
            } else {
                await fetch("/trigger_capture", { method: "POST" });
            }
        } catch (e) {
            console.error(e);
            output.textContent = "發生錯誤";
        }
    });

    // 輪詢最新描述
    async function pollLatestDesc() {
        try {
            const res = await fetch("/latest_desc");
            const text = await res.text();
            if (text && text.trim() !== "") {
                output.textContent = text;
            }
        } catch (e) {
            console.error(e);
        }
        setTimeout(pollLatestDesc, 500);
    }

    pollLatestDesc();
});
</script>
</div>
</body>
</html>
"""
@app.route('/debug_push', methods=['GET', 'POST'])
def debug_push():
    with chat_lock:
        chat_history.append(["(debug)", "這是一行測試文字"])
    return "OK"

@app.route('/auto_capture', methods=['POST'])
def auto_capture():
    with frame_lock:
        frame_copy = frame_shared.copy() if frame_shared is not None else None

    if frame_copy is None:
        return "(無可拍攝畫面)", 400

    auto_capture_on_hand()
    return "(自動倒數拍照中)"

@app.route('/trigger_capture', methods=['POST'])
def trigger_capture():
    pil_image = None
    with frame_clean_lock:
        if frame_clean is not None:
            frame_copy = frame_clean.copy()
            pil_image = save_frame_and_get_pil(frame_copy)

    # DEBUG log
    print("[DEBUG] trigger_capture called")
    print("[DEBUG] MODEL_READY =", MODEL_READY)
    print("[DEBUG] frame_clean is None =", frame_clean is None)
    print("[DEBUG] pil_image is None =", pil_image is None)

    # 先加入 chat_history 再顯示「生成中…」
    with chat_lock:
        chat_history.append(["(手動拍照)", "生成中…"])

    if not MODEL_READY:
        return "(模型尚未載入完成，請稍候…)", 503

    if pil_image:
        print("[DEBUG] start_image_description triggered")
        start_image_description(pil_image, source="(手動拍照)")
    else:
        # 若沒有 frame，也啟動「等待 frame 生成描述」的 thread
        print("[DEBUG] 無可用畫面，將等待下一個 frame 生成描述")
        def wait_for_frame_and_generate():
            timeout = 5.0  # 最多等 5 秒
            interval = 0.05
            waited = 0.0
            img_to_use = None
            while waited < timeout:
                with frame_clean_lock:
                    if frame_clean is not None:
                        img_to_use = save_frame_and_get_pil(frame_clean.copy())
                        break
                time.sleep(interval)
                waited += interval
            if img_to_use:
                start_image_description(img_to_use, source="(手動拍照等待)")
            else:
                print("[DEBUG] 超時，沒有 frame 可以生成描述")
                with chat_lock:
                    chat_history[-1][1] = "無可用畫面，生成失敗"

        threading.Thread(target=wait_for_frame_and_generate, daemon=True).start()

    return "(拍照請求已送出，生成中…)"

@app.route("/")
def index():
    # 每次進首頁播放比讚音
    play_startup_sound()

    with chat_lock:
        ch_copy = list(chat_history)

    status_text = "系統啟動中..." if not MODEL_READY else "系統啟動完成"
    return render_template_string(
        HTML_TEMPLATE.replace("系統啟動中...", status_text),
        chat=ch_copy
    )

@app.route('/hand_status')
def hand_status():
    global last_good_time
    # 只要最近 2 秒內有 GOOD，就回傳 True
    if (time.time() - last_good_time) < 2.0:
        return jsonify({"good": True})
    return jsonify({"good": False})

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/latest_desc')
def get_latest_desc():
    with latest_desc_lock:
        return latest_desc

@app.route('/upload_image', methods=['POST'])
def upload_image():
    global latest_desc

    file = request.files.get('file')
    if not file:
        return "沒有收到圖片"

    image = Image.open(file.stream).convert("RGB")

    start_image_description(image, source="(上傳圖片)")

    return "圖片已接收，開始生成"

# ---------------- gen_frames 手勢判斷 ----------------
def gen_frames():
    global frame_shared, frame_clean
    global hand_good_detected, last_good_time, good_cooldown

    good_counter = 0
    GOOD_THRESHOLD = 6   
    w, h = 700, 500

    def vector_2d_angle(v1, v2):
        try:
            return math.degrees(
                math.acos(
                    (v1[0]*v2[0] + v1[1]*v2[1]) /
                    (((v1[0]**2 + v1[1]**2) ** 0.5) *
                     ((v2[0]**2 + v2[1]**2) ** 0.5))
                )
            )
        except Exception:
            return 180

    def hand_angle(hand_):
        return [
            vector_2d_angle((hand_[0][0]-hand_[2][0], hand_[0][1]-hand_[2][1]),
                            (hand_[3][0]-hand_[4][0], hand_[3][1]-hand_[4][1])),
            vector_2d_angle((hand_[0][0]-hand_[6][0], hand_[0][1]-hand_[6][1]),
                            (hand_[7][0]-hand_[8][0], hand_[7][1]-hand_[8][1])),
            vector_2d_angle((hand_[0][0]-hand_[10][0], hand_[0][1]-hand_[10][1]),
                            (hand_[11][0]-hand_[12][0], hand_[11][1]-hand_[12][1])),
            vector_2d_angle((hand_[0][0]-hand_[14][0], hand_[0][1]-hand_[14][1]),
                            (hand_[15][0]-hand_[16][0], hand_[15][1]-hand_[16][1])),
            vector_2d_angle((hand_[0][0]-hand_[18][0], hand_[0][1]-hand_[18][1]),
                            (hand_[19][0]-hand_[20][0], hand_[19][1]-hand_[20][1])),
        ]

    def hand_pos(angles):
        f1, f2, f3, f4, f5 = angles
        if f1 < 50 and all(a >= 50 for a in [f2, f3, f4, f5]):
            return "good"
        if f3 < 50 and all(a >= 50 for a in [f1, f2, f4, f5]):
            return "no!!!"
        return ""

    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, img = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        img = cv2.resize(img, (w, h))
        clean_img = img.copy()

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        gesture = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                pts = [(lm.x*w, lm.y*h) for lm in hand_landmarks.landmark]
                fx = [int(x) for x, y in pts]
                fy = [int(y) for x, y in pts]

                angles = hand_angle(pts)
                gesture = hand_pos(angles)

                if gesture == "no!!!":
                    x1, x2 = max(min(fx)-10, 0), min(max(fx), w-1)
                    y1, y2 = max(min(fy)-10, 0), min(max(fy), h-1)
                    mosaic = img[y1:y2, x1:x2]
                    if mosaic.size:
                        mosaic = cv2.resize(mosaic, (8, 8))
                        mosaic = cv2.resize(mosaic, (x2-x1, y2-y1),
                                             interpolation=cv2.INTER_NEAREST)
                        img[y1:y2, x1:x2] = mosaic
                else:
                    cv2.putText(img, gesture, (30, 120),
                                font, 5, (255, 255, 255), 10, cv2.LINE_AA)

        # ---------- GOOD 累積 ----------
        if gesture == "good":
            good_counter += 1
        else:
            good_counter = 0

        if good_counter >= GOOD_THRESHOLD and not good_cooldown:
            good_cooldown = True
            good_counter = 0

            hand_good_detected = True
            last_good_time = time.time()

            auto_capture_on_hand()

        # ---------- 更新共享畫面 ----------
        with frame_clean_lock:
            frame_clean = clean_img.copy()

        with frame_lock:
            frame_shared = img.copy()

        ret, buffer = cv2.imencode(".jpg", img)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

# ---------------- 啟動 ----------------
if __name__ == "__main__":
    try:
        print("[startup] 啟動背景模型載入...")
        threading.Thread(target=lazy_load_qwen_bg, daemon=True).start()
        print("[startup] 啟動 Flask 伺服器")
        app.run(host="127.0.0.1", port=PORT, threaded=True, debug=False) #需要改成自己的IP
    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()