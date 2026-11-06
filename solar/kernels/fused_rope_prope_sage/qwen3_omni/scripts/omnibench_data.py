"""OmniBench loading + Qwen3-Omni prompt construction (shared by smoke test and benchmark)."""
import glob, io
import numpy as np, pyarrow.parquet as pq, soundfile as sf, librosa
from PIL import Image

PARQUETS = sorted(glob.glob("/home/gta/.cache/huggingface/hub/datasets--m-a-p--OmniBench/snapshots/*/data/*.parquet"))
LETTERS = "ABCD"

def iter_samples(limit=None):
    n = 0
    for f in PARQUETS:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=16):
            for row in batch.to_pylist():
                yield decode(row)
                n += 1
                if limit and n >= limit:
                    return

def decode(row):
    audio, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
    opts = row["options"]
    return dict(index=row["index"], task=row["task type"], audio_type=row["audio type"],
                question=row["question"], options=opts, answer=LETTERS[opts.index(row["answer"])],
                audio=audio, image=image)

def build_inputs(processor, s, device, audio_repeat=1, max_pixels=None, extra_text_tokens=0):
    """Chat-template the sample. audio_repeat / max_pixels / extra_text_tokens stretch the
    prefill for the sequence-length sweep without changing the task."""
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(s["options"]))
    text = (f"Please answer the following question based on the given image and audio.\n"
            f"Question: {s['question']}\nOptions:\n{opts}\n"
            f"Answer with the option's letter from the given choices directly.")
    if extra_text_tokens:
        text = ("Context (ignore): " + "lorem ipsum " * (extra_text_tokens // 2) + "\n") + text
    audio = np.tile(s["audio"], audio_repeat) if audio_repeat > 1 else s["audio"]
    conv = [{"role": "user", "content": [{"type": "image", "image": s["image"]},
                                         {"type": "audio", "audio": audio},
                                         {"type": "text", "text": text}]}]
    prompt = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    kw = {}
    if max_pixels:
        kw["images_kwargs"] = {"max_pixels": max_pixels}
    inputs = processor(text=prompt, audio=[audio], images=[s["image"]], return_tensors="pt",
                       padding=True, use_audio_in_video=False, **kw)
    return inputs.to(device)
