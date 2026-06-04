import os
import json
import torch
from transformers import pipeline
import soundfile as sf
import librosa
import numpy as np

# Paths
audio_path = "d:/src/NPL/OmniVoice/0525.wav"
output_dir = "d:/src/NPL/OmniVoice/data/my_voice/wavs"
os.makedirs(output_dir, exist_ok=True)

print("Loading Whisper pipeline...")
# Check if GPU is available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Using whisper-base for fast local execution on GTX 1650 (around 140M parameters)
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
    chunk_length_s=30,
    device=device,
    return_timestamps=True
)

print("Loading audio for transcription...")
# Load audio at 16kHz for Whisper
audio_data, sr = librosa.load(audio_path, sr=16000)

print("Transcribing with Whisper (this may take a couple of minutes)...")
# Force language to Vietnamese since the video/user is Vietnamese
result = pipe(audio_data, generate_kwargs={"language": "vietnamese"})

# Reload audio at 24kHz for OmniVoice chunks
print("Loading audio at 24kHz for chunking...")
audio_24k, sr_24k = librosa.load(audio_path, sr=24000)

chunks = []
segments = result.get("chunks", [])
print(f"Found {len(segments)} segments from Whisper.")

for idx, segment in enumerate(segments):
    text = segment.get("text", "").strip()
    timestamps = segment.get("timestamp")
    if not text or not timestamps:
        continue
    
    start_s, end_s = timestamps
    if start_s is None or end_s is None:
        continue
        
    duration = end_s - start_s
    # Filter out chunks that are too short or too long
    if duration < 1.5 or duration > 15.0:
        continue
        
    start_idx = int(start_s * sr_24k)
    end_idx = int(end_s * sr_24k)
    
    chunk_audio = audio_24k[start_idx:end_idx]
    chunk_name = f"chunk_{idx:04d}.wav"
    chunk_path = os.path.join(output_dir, chunk_name)
    
    # Save chunk at 24kHz
    sf.write(chunk_path, chunk_audio, sr_24k)
    
    chunks.append({
        "id": f"my_voice_{idx:04d}",
        "audio_path": os.path.abspath(chunk_path),
        "text": text,
        "language_id": "vi"
    })

print(f"Successfully processed {len(chunks)} chunks.")

# Split into train (95%) and dev (5%)
np.random.seed(42)
np.random.shuffle(chunks)
split_idx = int(len(chunks) * 0.95)
train_chunks = chunks[:split_idx]
dev_chunks = chunks[split_idx:]

# Write train JSONL
train_jsonl_path = "d:/src/NPL/OmniVoice/data/my_data_train.jsonl"
with open(train_jsonl_path, "w", encoding="utf-8") as f:
    for c in train_chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

# Write dev JSONL
dev_jsonl_path = "d:/src/NPL/OmniVoice/data/my_data_dev.jsonl"
with open(dev_jsonl_path, "w", encoding="utf-8") as f:
    for c in dev_chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

print(f"Manifests successfully written:")
print(f" - Train: {train_jsonl_path} ({len(train_chunks)} samples)")
print(f" - Dev:   {dev_jsonl_path} ({len(dev_chunks)} samples)")
