import os
import time
import uuid
import asyncio
import gradio as gr
import torch
import opencc
from PIL import Image, ImageOps
from typing import List, Optional, Any
import threading

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    TextIteratorStreamer,
)

def normalize_china(text: str):
    return text.replace("中國", "中國大陸")

def cut_to_user_question(user_text: str, model_output: str, has_image: bool = False) -> str:
    # 圖片模式不要裁切
    return model_output

    
def to_traditional(text: str):
    """調用 OpenCC，把輸入轉為繁體"""
    try:
        return converter.convert(text)
    except:
        return text
from faster_whisper import WhisperModel
import edge_tts
# -------------------- 嚴格簡體偵測 --------------------
# 常見簡體字（2000+ 字也可給你完整表，目前給核心常見高風險字）
SIMPLIFIED_SET = set("万与丑专着业丰为举么义乌乐乔习乡边书买乱争亏云亘亚产亩亲亵仆仅从仑仓仪们众优伙会伞伟传伤伥伦伧伪伫体余佣佥侠侣侥侦侧侨侩侪侬俭债倾偬偻偿傥倾传债伤伥伦伧佥侥侦侧侨侩侪俨俩俪俫俬俭债储")

import re
def contains_simplified(text: str) -> bool:
    if not text:
        return False
    # 若文字中包含任一簡體字 → 判定為簡體
    return any(ch in SIMPLIFIED_SET for ch in text)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 顯示 transformers 詳細日誌，方便診斷不支援的 generation 參數
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'info')

# -------------------- USER CONFIG --------------------
MODEL_PATH = r"D:\DATASET"
LOGO_PATH = r"D:\未命名.png"
PORT = 7860

HAS_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if HAS_CUDA else "cpu"

# -------------------- OpenCC --------------------
converter = opencc.OpenCC("s2t.json")


def to_traditional(s):
    try:
        return converter.convert(s)
    except Exception:
        return s


# -------------------- LOAD MODEL --------------------
print("Loading DATASET...")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16 if HAS_CUDA else torch.float32,
    device_map="auto" if HAS_CUDA else None,
    trust_remote_code=True,
)

try:
    model.config.use_cache = True
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = True
except Exception:
    pass

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

tokenizer: PreTrainedTokenizerBase = getattr(processor, "tokenizer", None)
if tokenizer is None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("Model ready.")

strict_chat_template = """
{% for message in messages %}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}
<|im_end|>
{% endfor %}
<|im_start|>assistant
"""

tokenizer.chat_template = strict_chat_template
processor.tokenizer.chat_template = strict_chat_template


# -------------------- Whisper STT --------------------
whisper_model = WhisperModel("small", device=DEVICE)


def stt_from_file_sync(path):
    if not path:
        return ""
    seg, _ = whisper_model.transcribe(path)
    return "".join([s.text for s in seg])


# -------------------- Edge-TTS --------------------
async def synthesize_edge(text, out_path):
    comm = edge_tts.Communicate(text, "zh-TW-HsiaoChenNeural")
    await comm.save(out_path)
    return out_path


def synthesize_edge_sync(text, out_path=None):
    """
    同步 wrapper：
    - 若沒有正在運行的 event loop，直接 asyncio.run()
    - 若已有運行中的 loop（例如 Gradio），則在新執行緒建立新的 loop 並執行 coroutine
    這樣可以避免在已運行的 loop 使用 asyncio.run() 導致錯誤。
    """
    if not out_path:
        out_path = f"tts_{int(time.time())}_{uuid.uuid4().hex}.mp3"

    # 嘗試取得目前 loop；部分環境會噴錯，用 try/except 包住
    try:
        loop = asyncio.get_event_loop()
    except Exception:
        loop = None

    # 若沒有 loop 或 loop 未在運行，直接 asyncio.run
    if loop is None or not loop.is_running():
        return asyncio.run(synthesize_edge(text, out_path))

    # 若已有正在運行的 loop（常見於 Gradio），在新執行緒建立新的 loop 執行 coroutine
    result = {"out": None, "exc": None}

    def _runner():
        try:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            new_loop.run_until_complete(synthesize_edge(text, out_path))
            new_loop.close()
            result["out"] = out_path
        except Exception as e:
            result["exc"] = e

    t = threading.Thread(target=_runner)
    t.start()
    t.join()

    if result["exc"]:
        raise result["exc"]
    return result["out"]


# -------------------- Helpers --------------------
SYSTEM_PROMPT = (
    "使用者輸入中文時，務必使用繁體中文回覆；若使用者使用英文，則以英文回覆。"
    "任何簡體字都必須轉為繁體後再輸出。"
    "若使用者詢問法律問題，嚴禁出現'法律依據參考（中國大陸大陸範圍)'等字樣。"
    "若使用者叫你生成程式碼，嚴禁生成程式碼以外的。"
)


def detect_language(text):
    if not text:
        return "zh"
    a = sum(1 for c in text if ord(c) < 128)
    return "en" if a / max(len(text), 1) > 0.6 else "zh"


# ---------- 圖片正規化 helper ----------
def normalize_image(img: Image.Image, max_side: int = 1280) -> Image.Image:
    """
    - 轉 RGB
    - 處理 EXIF 方向
    - 若圖片太大，依比例縮小（避免 GPU memory 爆掉）
    """
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    if img.mode != "RGB":
        img = img.convert("RGB")
    # 縮放（若任一邊大於 max_side，等比縮）
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img


# ----------------------------------------------------
#               Qwen3-VL 推論核心（整合版）
# ----------------------------------------------------

def _build_messages(user_text: str, images: List[Image.Image], history: List[List[str]]):
    """
    建立 messages list：
    - system message
    - 跟隨歷史（每一個 turn 期望 history 為 [user, assistant]）
    - 最後一個 user message，內容可包含多張 image 與 text
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    text_prompt = processor.apply_chat_template(messages, tokenize=False)
    
    if history:
        # history 是 list of [user, assistant]
        for turn in history:
            if not isinstance(turn, (list, tuple)) or len(turn) < 1:
                continue
            user_msg = turn[0] if len(turn) > 0 else ""
            assistant_msg = turn[1] if len(turn) > 1 else ""

            # user
            messages.append({"role": "user", "content": user_msg})
            # assistant (如果有)
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})

    # build content for current user (images + text)
    content = []
    for img in images or []:
        # ensure PIL Image and normalize
        try:
            nimg = normalize_image(img)
        except Exception:
            nimg = img
        content.append({"type": "image", "image": nimg})
    content.append({"type": "text", "text": user_text or ""})

    messages.append({"role": "user", "content": content})
    return messages


def _processor_prepare_inputs(messages):
    """
    嘗試以較新方法轉換 messages（apply_chat_template），若 processor 沒有該方法則 fallback 到
    processor(messages=...)。最後回傳 tensor dict 並移到 model.device。

    為了避免 image-token mismatch，當 messages 含 image 時：
    1) 用 apply_chat_template(..., tokenize=False) 取得 prompt_text
    2) 再呼叫 processor(text=prompt_text, images=imgs, return_tensors="pt")
    """
    # 檢查 messages 中是否含 image
    imgs = []
    try:
        last = messages[-1]["content"]
        for c in last:
            if isinstance(c, dict) and c.get("type") == "image":
                img_obj = c.get("image")
                if img_obj is not None:
                    try:
                        img_obj = normalize_image(img_obj)
                    except Exception:
                        pass
                    imgs.append(img_obj)
    except Exception:
        imgs = []

    # Note: 不同版本的 processor API 不同，這裡採用 try/except 容錯
    try:
        # 若包含 image，我們先用 tokenize=False 取得 prompt_text，再把 text+images 一起傳給 processor
        if imgs:
            prompt_text = processor.apply_chat_template(messages, tokenize=False)
            inputs = processor(text=prompt_text, images=imgs, return_tensors="pt")
        else:
            # 沒有 images 時直接用 tokenize=True 的快捷方式
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt"
            )
    except Exception:
        # fallback: 有些版本支援直接傳 messages
        try:
            # 正確：先將 messages 轉成文字模板
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt"
            ).to(model.device)

        except TypeError:
            # 再 fallback：用 text/images 分開
            # 嘗試把最後 user content 拆回 text & images
            try:
                last = messages[-1]["content"]
                imgs2 = [c["image"] for c in last if isinstance(c, dict) and c.get("type") == "image"]
                texts = [c["text"] for c in last if isinstance(c, dict) and c.get("type") == "text"]
                text = texts[0] if texts else ""
                if imgs2:
                    imgs2 = [normalize_image(i) for i in imgs2]
                inputs = processor(
                    images=imgs2 if imgs2 else None,
                    text=text,
                    padding="max_length",
                    max_length=512,
                    return_tensors="pt"
                )
            except Exception:
                # 最後跌回到把所有 messages 當成 text
                prompt_text = ""
                try:
                    prompt_text = processor.apply_chat_template(messages, tokenize=False)
                except Exception:
                    prompt_text = str(messages)
                inputs = processor(text=prompt_text, return_tensors="pt")

    # move tensors to model device if returned as dict of tensors
    if isinstance(inputs, dict):
        for k, v in list(inputs.items()):
            try:
                if hasattr(v, "to"):
                    inputs[k] = v.to(model.device)
            except Exception:
                pass
    return inputs


def safe_model_generate(model, *args, **kwargs):
    """Robust wrapper for model.generate.

    - Removes unsupported generation kwargs ('temperature','top_p','top_k').
    - Recursively moves any torch.Tensor in args/kwargs to the model device
      to avoid device-mismatch RuntimeError.
    - On device-related RuntimeError, retries once after forcing device move.
    """
    # filter out unsupported/invalid keys
    for k in ("temperature", "top_p", "top_k"):
        if k in kwargs:
            kwargs.pop(k)

    def _move_to_device(obj, device):
        # move tensors recursively; leave other objects untouched
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

    # determine model device
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
        # if device mismatch or image-token mismatch, try one more time after forcing move
        if ('Expected all tensors to be on the same device' in msg) or ('Image features and image tokens do not match' in msg):
            try:
                # recompute device from model and move again
                try:
                    model_device = next(model.parameters()).device
                except Exception:
                    model_device = getattr(model, 'device', model_device)
                new_args = tuple(_move_to_device(a, model_device) for a in new_args)
                new_kwargs = {k: _move_to_device(v, model_device) for k, v in new_kwargs.items()}
                return model.generate(*new_args, **new_kwargs)
            except Exception:
                # fall through to raise original
                pass
        raise


def run_model(user_text: str, images: List[Image.Image], history):
    # 使用 message-builder + processor wrapper 產生正確的 inputs
    messages = _build_messages(user_text, images or [], history)
    inputs = _processor_prepare_inputs(messages)

    # 呼叫 generate，若 model 或 processor 有相容性問題，會由 safe_model_generate 過濾不支援的生成參數
    output = safe_model_generate(
        model,
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        use_cache=True
    )

    # output 可能是 tensor of shape (1, seq_len)
    if isinstance(output, torch.Tensor):
        seq = output[0]
    else:
        try:
            seq = output[0]
        except Exception:
            seq = output

    try:
        text = tokenizer.decode(seq, skip_special_tokens=True)
    except Exception:
        # 若 decode 失敗，嘗試轉成字串
        try:
            if isinstance(seq, torch.Tensor):
                seq_list = seq.cpu().tolist()
                text = tokenizer.decode(seq_list, skip_special_tokens=True)
            else:
                text = str(seq)
        except Exception:
            text = str(seq)

    # 移除 prompt
    if "Assistant:" in text:
        text = text.split("Assistant:")[-1].strip()

    return to_traditional(text)


def stream_model(user_text: str, images: List[Image.Image], history): 
    # ==== 組 Prompt & 使用 _processor_prepare_inputs 保證圖片被包含 ====
    msgs = _build_messages(user_text, images or [], history)
    inputs = _processor_prepare_inputs(msgs)

    # ==== Streamer ====
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_special_tokens=True,
        skip_prompt=True,
    )

    def gen():
        safe_model_generate(
            model,
            **inputs,
            max_new_tokens=150,
            repetition_penalty=1.2,  # 降低模型重複擴寫傾向
            do_sample=False,
            no_repeat_ngram_size=3,        # 阻止模型重複片語拖長文本
            streamer=streamer,
            use_cache=True
        )

    threading.Thread(target=gen).start()

    buffer = ""
    for chunk in streamer:
        buffer += chunk

    # ====== 新增：只允許與使用者問題相關的輸出 ======
        cleaned = cut_to_user_question(user_text, buffer, has_image=bool(images))
        cleaned = to_traditional(cleaned)
        cleaned = normalize_china(cleaned)

        if "\n" in buffer:
            yield cleaned
            buffer = ""


    # ====== 還有殘留字串 → 最後一次輸出 ======
    if buffer:
        buffer = to_traditional(buffer)
        buffer = normalize_china(buffer)
        yield buffer

# ----------------------------------------------------
#                     Gradio UI
# ----------------------------------------------------
css = """
:root { --bg:#0b1020 }
body { background:var(--bg); color:#e6eef6 }
#chat-area { height:740px }
"""

with gr.Blocks(title="DeepChat", css=css) as demo:

    gr.Markdown("<h2 style='text-align:center;color:white'>DeepChat</h2>")

    chatbot = gr.Chatbot(label="對話紀錄", elem_id="chat-area")

    with gr.Row():
        input_box = gr.Textbox(show_label=False, placeholder="輸入訊息…")
        image_input = gr.File(
            label="上傳圖片（可多張）",
            file_types=["image"],
            file_count="multiple"
        )

    mic_btn = gr.Audio(sources=["microphone"], type="filepath", label="🎤")
    send_btn = gr.Button("送出")

    with gr.Row():
        clear_btn = gr.Button("清除對話")
        tts_btn = gr.Button("語音播出")

    # ------------ Submit（generator，支援 streaming） ------------
    def _open_file_to_pil(f: Any):
        # f 可能是 str path / dict / tempfile-like
        if f is None:
            return None
        try:
            if isinstance(f, str):
                img = Image.open(f)
                return normalize_image(img)
            # gradio may give a dict with 'name' or 'file'
            if isinstance(f, dict):
                path = f.get("name") or f.get("tmp_path") or f.get("file")
                if path:
                    img = Image.open(path)
                    return normalize_image(img)
            # file-like object
            if hasattr(f, "name"):
                img = Image.open(f.name)
                return normalize_image(img)
        except Exception:
            return None
        return None

    def submit_fn(user_text, user_images, chat_history):
                # ---------- 強制繁體機制（不動你任何圖片邏輯） ----------
        if contains_simplified(user_text):
            user_text = to_traditional(user_text)

        # ---- 保護：確保 chat_history 不會被 streaming 汙染 ----
        if chat_history is None:
            chat_history = []
        else:
            chat_history = [[u, a] for (u, a) in chat_history]

        # 新增使用者訊息
        new_item = [user_text, ""]
        chat_history = chat_history + [new_item]

        yield chat_history, "", None

        # ---- 完全不動你圖片處理邏輯 ----
        pil_imgs = []
        if user_images:
            imgs = user_images if isinstance(user_images, (list, tuple)) else [user_images]
            for f in imgs:
                pil = _open_file_to_pil(f)  # 保留原本你的處理
                if pil is not None:
                    pil_imgs.append(pil)

        # ---- local buffer（不污染 history）----
        assistant_text = ""

        try:
            # 關鍵：不把尚未填滿的最後一行 history 傳進 streaming
            for chunk in stream_model(user_text, pil_imgs, chat_history[:-1]):
                assistant_text += chunk
                chat_history[-1][1] = assistant_text
                yield chat_history, "", None

        except Exception as e:
            err = f"(推論失敗：{e})"
            chat_history[-1][1] = err
            yield chat_history, "", None
            return

        # 最終產物填入
        chat_history[-1][1] = assistant_text
        yield chat_history, "", None

    # 綁定按鈕與 Enter（必須在 with 範圍內）
    send_btn.click(
        submit_fn,
        inputs=[input_box, image_input, chatbot],
        outputs=[chatbot, input_box, image_input]
    )

    input_box.submit(
        submit_fn,
        inputs=[input_box, image_input, chatbot],
        outputs=[chatbot, input_box, image_input]
    )

    # ------------ Mic → STT → Auto send ------------
    def mic_to_text(fp, history):
        if history is None:
            history = []

        if not fp:
            return history, ""

        text = to_traditional(stt_from_file_sync(fp))

        # 直接用 run_model（非 streaming）
        reply = run_model(text, [], history)

        history = history + [[text, reply]]
        return history, ""

    mic_btn.change(
        mic_to_text,
        inputs=[mic_btn, chatbot],
        outputs=[chatbot, input_box]
    )

    # ------------ TTS ------------
    def tts_play(history):
        if not history:
            return None
        # history item 是 [user, assistant]
        last = history[-1]
        if isinstance(last, (list, tuple)) and len(last) > 1:
            msg = last[1]
        elif isinstance(last, dict):
            msg = last.get("content", "")
        else:
            msg = ""
        out = f"tts_{time.time()}.mp3"
        return synthesize_edge_sync(msg, out)

    tts_btn.click(
        tts_play,
        inputs=[chatbot],
        outputs=gr.Audio(type="filepath")
    )

    # ------------ Clear ------------
    clear_btn.click(lambda: [], None, chatbot)

# -------------------- Launch --------------------
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=PORT, share=False)
