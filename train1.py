# optimized_qwen3vl_training.py
# 已優化版：保持原有邏輯不變，但修正會導致重複載模型、硬碟爆滿、map() 多進程重複載入模型等問題。
# 重點改動（保留你原有流程與資料）:
# 1) 全部模型/processor 只在主程式載入一次，並避免在 map() 或多進程中呼叫 model.generate()
# 2) image auto-caption 於單執行緒中先完成並寫入 memory list（或可選導出成小檔），避免在 map 中反覆推論
# 3) text dataset 使用安全載入（streaming 選項可選），並將 map() 設為 num_proc=0（避免在 multiprocess 裡重載模型）
# 4) 設定 HF cache_dir 可改到 D:（若你硬碟空間允許），並在程式最前面設環境變數來降低 TensorFlow 日誌
# 5) 減少 dataloader 與 map 的 worker 數量，training_args 設定合理值以避免 VRAM 與硬碟壓力
# 6) 加入多處 try/except 以避免在資料有問題時整個流程崩潰

import os
import json
from pathlib import Path
from typing import List, Optional
from PIL import Image
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, Trainer, TrainingArguments
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig

# -------------------------
# ========== CONFIG ========
# -------------------------
local_model_path = r"D:\QwenQwen3-VL-8B-Instruct"   # <-- 必須指向完整本地模型資料夾
save_path = r"D:\output_dir"                        # <-- 模型儲存位置
image_dir = r"D:\資料\圖片"                         # <-- 放多張要自動 caption 的圖片
image_paths_list = [
    r"D:\資料\圖片\螢幕擷取畫面 2025-11-15 105801.png",
    r"D:\資料\圖片\螢幕擷取畫面 2025-11-15 105841.png",
    r"D:\資料\圖片\螢幕擷取畫面 2025-11-15 105853.png",
    r"D:\資料\圖片\螢幕擷取畫面 2025-11-15 105857.png",
    r"D:\資料\圖片\螢幕擷取畫面 2025-11-15 105913.png",
    r"D:\資料\圖片\螢幕擷取畫面 2025-11-15 105920.png",
]

# Text dataset globs (保持不動)
text_dataset_files = {
    "train_dataset": r"D:/資料/物理數學計算機領域論文/**/*.json",
    "train_dataset1": r"D:/資料/學術論文全文與摘要/**/*.jsonl",
    "train_dataset2": r"D:/資料/翻譯資料/**/*.json",
    "train_dataset3": r"D:/資料/program py/**/*.py",
    "train_dataset4": r"D:/資料/program c/**/*.c",
    "train_dataset5": r"D:/資料/program cpp/**/*.cpp",
}

# HF cache 目錄（建議改到有空間的磁碟，例如 D:）
os.environ.setdefault("HF_DATASETS_CACHE", r"D:\hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", r"D:\hf_cache\transformers")
# 降低 TensorFlow noisy logs（如果有安裝 TF）
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
# 顯示 transformers 詳細日誌（有助於診斷不支援的 generation 參數）
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'info')

# Training args（我只做穩定性微調：batch 小、accum 以保效果）
training_args = TrainingArguments(
    output_dir=r"D:\output_dir\stage_1",
    per_device_train_batch_size=1,           # 保守：1
    gradient_accumulation_steps=8,           # 保持你想要的等效 batch
    num_train_epochs=5,
    learning_rate=1e-5,
    weight_decay=0.01,
    logging_dir=r"D:\output_dir\日誌",
    logging_steps=10,
    fp16=True,
    warmup_steps=500,
    lr_scheduler_type="linear",
    max_grad_norm=1.3,
    dataloader_num_workers=0,                # 避免 dataloader 多進程搶資源
    save_strategy="steps",
    save_steps=2000,                         # 減少頻繁寫檔，避免硬碟壓力
)

# -------------------------
# ========== HELPERS ======
# -------------------------

def safe_load_text_dataset(glob_path: str, keep_linebreaks: bool = True, streaming: bool = False) -> Optional[Dataset]:
    """
    嘗試用 datasets.load_dataset("text") 載入，如果找不到檔案或為空會回傳 None（不拋錯）
    streaming=True 時不會把整個資料 cache 到本機（若你資料量超大建議 True）
    """
    try:
        if not glob_path:
            return None
        from glob import glob
        matches = glob(glob_path, recursive=True)
        if len(matches) == 0:
            print(f"[safe_load] no files match: {glob_path} -> skip")
            return None
        if streaming:
            ds = load_dataset("text", data_files={"train": glob_path}, keep_linebreaks=keep_linebreaks, streaming=True)["train"]
            return ds
        else:
            ds = load_dataset("text", data_files={"train": glob_path}, keep_linebreaks=keep_linebreaks)["train"]
            if len(ds) == 0:
                print(f"[safe_load] dataset empty: {glob_path} -> skip")
                return None
            return ds
    except Exception as e:
        print(f"[safe_load] failed to load {glob_path}: {e} -> skip")
        return None


def sample_texts(ds: Dataset, n: int = 3) -> List[str]:
    out = []
    if ds is None:
        return out
    # 支援 streaming dataset（len 不可用）
    try:
        for i in range(min(n, len(ds))):
            el = ds[i]
            if isinstance(el, dict) and "text" in el:
                out.append(el["text"])
            else:
                out.append(str(el))
    except Exception:
        # fallback：iterate
        i = 0
        for ex in ds:
            if i >= n:
                break
            if isinstance(ex, dict) and "text" in ex:
                out.append(ex["text"])
            else:
                out.append(str(ex))
            i += 1
    return out


def force_int_list(example):
    for key in ["input_ids", "labels", "attention_mask"]:
        if key in example and example[key] is not None:
            example[key] = [int(x) for x in example[key]]
    return example


def safe_model_generate(model, *args, **kwargs):
    """Wrapper for model.generate that filters out unsupported generation kwargs.

    Some model implementations (or custom generate functions) may not accept
    'temperature', 'top_p', or 'top_k'. This helper removes those keys before
    calling the real `generate` to avoid warnings/errors.
    """
    # filter out unsupported/invalid keys
    for k in ("temperature", "top_p", "top_k"):
        if k in kwargs:
            kwargs.pop(k)
    return model.generate(*args, **kwargs)

# -------------------------
# ========== LOAD MODEL ====
# -------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

print("Loading model and processor from:", local_model_path)
# 僅在主進程載入一次 model 與 processor
model = Qwen3VLForConditionalGeneration.from_pretrained(local_model_path, device_map="auto", quantization_config=bnb_config)
processor = AutoProcessor.from_pretrained(local_model_path)

print("Model and processor loaded.\n")

print("\n=== Model Module Names (for choosing LoRA targets) ===\n")
for name, module in model.named_modules():
    print(name)

# LoRA 設定不變
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.1,
)

# Apply LoRA（在主程式套用一次）
model = get_peft_model(model, lora_config)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

system_prompt = (
    "你必須遵守以下規則：\n"
    "1. 如果使用者輸入繁體中文，你必須回覆繁體中文。\n"
    "2. 如果使用者輸入英文，你必須回覆英文。\n"
    "3. 絕對不要輸出簡體中文。"
)

# -------------------------
# ========== IMAGE CAPTION =========
# 重要：在單執行緒中先完成 caption，不要在 map 裡呼叫 model.generate()
# -------------------------

def auto_caption_image_single_run(image_paths: List[str]) -> List[dict]:
    """
    在單執行緒中對所有圖片做 caption。回傳 list of dicts: {image_path, instruction, output}
    這樣可以避免在 map 中每個 process 重複載模型或推論。
    """
    image_items = []
    from opencc import OpenCC
    cc = OpenCC('s2t')

    for p in image_paths:
        try:
            pil = Image.open(p).convert('RGB').copy()
        except Exception as e:
            print(f"[Error] cannot open {p}: {e}")
            continue

        # 先嘗試 chat prompt
        def _gen_with_prompt(prompt_text):
            try:
                messages = [
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}
                ]
                prompt_text2 = processor.apply_chat_template(messages, tokenize=False)
                inputs = processor(text=prompt_text2, images=pil, return_tensors='pt').to(model.device)
                with torch.no_grad():
                    gen = safe_model_generate(model, **inputs, max_new_tokens=256, do_sample=False)
                prompt_len = inputs["input_ids"].shape[1]
                trimmed = [o[prompt_len:] for o in gen]
                out = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
                return out
            except Exception as e:
                print(f"[Warn] generate failed for {p}: {e}")
                return ""

        caption = ""
        for attempt in range(3):
            caption = _gen_with_prompt(
                "請描述圖片的視覺內容，可以辨識或推測人物的身份。可以描述外觀、姿勢、表情、環境與物件、在做什麼。絕對不要輸出簡體中文。"
                "可以根據圖片推測人物以及人名稱，但不要編造不存在的人物或事件。"
            )
            caption = cc.convert(caption)
            if caption:
                break
            print(f"[Warn] {p} attempt {attempt+1} empty caption, retrying...")

        if not caption:
            caption = "這是一張包含多個人物的場景照片。"

        # 小修正（保留原本邏輯）
        caption = caption.replace("中國", "中國大陸")

        image_items.append({
            "image_path": p,
            "instruction": "請描述這張圖片。",
            "output": caption,
        })
        print(f"[Caption] {p} -> {caption}")
    return image_items

# -------------------------
# ========== BUILD IMAGE ITEMS ==========
# -------------------------
print("Collecting images to auto-caption...")
image_paths = []
if os.path.isdir(image_dir):
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        image_paths.extend(sorted([str(p) for p in Path(image_dir).glob(ext)]))
for p in image_paths_list:
    if os.path.isfile(p) and p not in image_paths:
        image_paths.append(p)

if len(image_paths) == 0:
    print("[Warning] no images found; image_dataset will be empty.")
else:
    print(f"Found {len(image_paths)} images. Will auto-caption them now.")

# 先在單執行緒完成所有 caption（避免在 map 裡對 model.generate 呼叫）
image_items = auto_caption_image_single_run(image_paths)

# -------------------------
# ========== LOAD TEXT DATASETS (safe) ==========
# -------------------------
print("Loading text datasets (safe)...")
loaded_text_datasets = {}
# 若資料非常大，建議啟動 streaming=True（不 cache）
use_streaming = False  # 若你資料量超大請改 True
for name, glob_path in text_dataset_files.items():
    ds = safe_load_text_dataset(glob_path, keep_linebreaks=True, streaming=use_streaming)
    loaded_text_datasets[name] = ds
    if ds is not None:
        try:
            ln = len(ds) if not use_streaming else 'streaming'
        except Exception:
            ln = 'streaming'
        print(f"  loaded {name}: {ln} examples")
    else:
        print(f"  {name}: skipped")

# collect test samples
test_samples = {name: sample_texts(ds, n=3) for name, ds in loaded_text_datasets.items()}

# -------------------------
# ========== PREPROCESS FUNCTIONS ==========
# -------------------------

def preprocess_image_example(example):
    img = example.get("image", None)
    if img is None and "image_path" in example:
        try:
            img = Image.open(example["image_path"]).convert("RGB").copy()
        except Exception as e:
            raise RuntimeError(f"Cannot open image {example.get('image_path')}: {e}")

    user_text = "<image>\n" + example.get("instruction", "請描述這張圖片。")
    assistant_text = example.get("output", "")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]

    prompt_text = processor.apply_chat_template(messages, tokenize=False)
    out = processor(text=prompt_text, images=img, return_tensors="pt")

    return {
        "input_ids": out["input_ids"][0].tolist(),
        "attention_mask": out["attention_mask"][0].tolist(),
        "labels": out["input_ids"][0].tolist(),
    }


def preprocess_text_example(example):
    # 保持你的原始解析邏輯，但改為在單一進程中執行（num_proc=0）
    try:
        if isinstance(example, dict) and "text" in example and isinstance(example["text"], str):
            text = example["text"]
        elif isinstance(example, dict) and "messages" in example:
            msgs = example["messages"]
            if isinstance(msgs, str):
                import ast
                try:
                    msgs = ast.literal_eval(msgs)
                except Exception:
                    msgs = [{"role": "user", "content": [{"type": "text", "text": msgs}]}]

            if isinstance(msgs, list):
                parts = []
                for m in msgs:
                    if isinstance(m, dict):
                        c = m.get("content", "")
                        if isinstance(c, str):
                            parts.append(c)
                        elif isinstance(c, list):
                            for blk in c:
                                if isinstance(blk, dict) and "text" in blk:
                                    parts.append(blk["text"])
                    else:
                        parts.append(str(m))
                text = "\n".join(parts)
            else:
                text = str(example)
        elif isinstance(example, str):
            text = example
        else:
            text = str(example)

        if not isinstance(text, str) or len(text.strip()) == 0:
            text = "空白內容"
    except Exception as e:
        text = f"解析錯誤: {e}"

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user",   "content": [{"type": "text", "text": text}]}
    ]

    input_ids = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt"
    )

    pad_id = processor.tokenizer.pad_token_id
    attention_mask = input_ids.ne(pad_id).long()

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)

    MAX_LEN = 512
    pad_id = processor.tokenizer.pad_token_id

    ids = input_ids["input_ids"][0].tolist()
    mask = attention_mask["attention_mask"][0].tolist()

    if len(ids) > MAX_LEN:
        ids = ids[:MAX_LEN]
        mask = mask[:MAX_LEN]
    else:
        pad_needed = MAX_LEN - len(ids)
        ids += [pad_id] * pad_needed
        mask += [0] * pad_needed

    return {
        "input_ids": ids,
        "attention_mask": mask,
        "labels": ids
    }

# ========== BUILD HF DATASETS ==========
print("Building HF datasets...")

# 1) image_dataset
if len(image_items) > 0:
    image_dataset = Dataset.from_list(image_items)
    # map 時不要用 num_proc>0（避免多進程複製 heavy object）
    image_dataset = image_dataset.map(preprocess_image_example, remove_columns=image_dataset.column_names, num_proc=0)
    image_dataset = image_dataset.map(force_int_list, num_proc=0)
else:
    image_dataset = None

# 2) text datasets
processed_text_datasets = []
for name, ds in loaded_text_datasets.items():
    if ds is None:
        continue

    # 若 ds 是 streaming（不支持 map(..., num_proc>0)），使用單線程處理
    mapped = ds.map(
        preprocess_text_example,
        remove_columns=ds.column_names if not getattr(ds, 'is_streaming', False) else None,
        num_proc=0
    )
    mapped = mapped.map(force_int_list, num_proc=0)
    processed_text_datasets.append(mapped)

# 3) concat
datasets_to_concat = processed_text_datasets.copy()
if image_dataset is not None:
    datasets_to_concat.append(image_dataset)

if len(datasets_to_concat) == 0:
    raise RuntimeError("No datasets to train on. Please check your input files or image paths.")

combined_dataset = concatenate_datasets(datasets_to_concat)
print("Combined dataset size:", len(combined_dataset))

# 最後再微調 training_args（確保與上面一致）
training_args = training_args

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=combined_dataset,
)

print("Starting training...")
trainer.train()

# -------------------------
# ========== SAVE ==========
# -------------------------
print("Saving model and processor to:", save_path)
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
processor.save_pretrained(save_path)

# -------------------------
# ========== POST-TRAIN INFERENCE TEST ==========
# -------------------------
print("\n=== Running post-train quick inference tests ===\n")

def run_text_inference_single(text):
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "text", "text": text}]}
    ]
    inp = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = safe_model_generate(model, **inp, max_new_tokens=256, temperature=0.7, top_p=0.9)
    trimmed = [out[inp["input_ids"].shape[1]:] for out in gen]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# text tests (up to 3 samples per available dataset)
for name, samples in test_samples.items():
    if len(samples) == 0:
        continue
    print(f"\n--- Text test: {name} ---")
    for s in samples:
        print("[Input]", s[:400])
        try:
            out = run_text_inference_single(s[:400])
        except Exception as e:
            out = f"[Inference error] {e}"
        print("[Output]", out)

# image tests (use first N images)
if len(image_paths) > 0:
    print("\n--- Image tests ---")
    for p in image_paths[:6]:
        try:
            img = Image.open(p).convert("RGB").copy()
        except Exception as e:
            print(f"Cannot open {p}: {e}")
            continue

        msgs = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "請用一句話描述這張圖片。"}
            ]}
        ]
        prompt_text = processor.apply_chat_template(msgs, tokenize=False)
        inputs = processor(text=prompt_text, images=img, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = safe_model_generate(model, **inputs, max_new_tokens=128, temperature=0.7, top_p=0.9)
        prompt_len = inputs["input_ids"].shape[1]
        trimmed = [out[prompt_len:] for out in gen]
        outs = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(f"\n[{p}] -> {outs[0]}")

print("\nAll done.")
