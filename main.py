import cv2
import torch
import numpy as np
import os
import imageio
from fastapi import FastAPI, UploadFile, File
from transformers import AutoImageProcessor, AutoModelForVideoClassification

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ckpt = "Shawon16/VideoMAE_Base_WLASL_100_200_epochs_p20_SR_8"
print(f"Starting AI engine, current device: {device.type.upper()}")
print("Downloading and loading model from Hugging Face (first startup may take several minutes, please wait patiently)...")
local_save_path = "./my_model_weights"

print(f"Preparing to download model, files will be safely saved in: {local_save_path} ...")

image_processor = AutoImageProcessor.from_pretrained(model_ckpt, cache_dir=local_save_path)
model = AutoModelForVideoClassification.from_pretrained(model_ckpt, cache_dir=local_save_path).to(device)
print("model loaded successfully, ready to serve")


def get_video_frames_fixed(video_path, num_frames=16):
    reader = imageio.get_reader(video_path, 'ffmpeg')
    all_frames = [frame for frame in reader]
    total_frames = len(all_frames)
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    selected_frames = []
    for idx in indices:
        frame = all_frames[idx]
        selected_frames.append(frame)
        
    return selected_frames

#API:
@app.get("/ping")
def ping_server():
    return {"status": "success", "message": f"running"}

@app.post("/translate")
async def translate_video(file: UploadFile = File(...)):
    temp_video_path = f"temp_{file.filename}"
    with open(temp_video_path, "wb") as buffer:
        buffer.write(await file.read())
        
    print(f"Received test video: {temp_video_path}")

    frames = get_video_frames_fixed(temp_video_path, num_frames=16)
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    test_frame = frames[8] 
    cv2.imwrite("./debug_mobile_frame.jpg", test_frame)
    if len(frames) == 0:
         return {"status": "error", "message": "unable to read video"}

    inputs = image_processor(list(frames), return_tensors="pt").to(device)

    with torch.no_grad(): 
        outputs = model(**inputs)
        logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()
    best_word = model.config.id2label[predicted_class_idx]
    
    confidence = torch.softmax(logits, dim=-1)[0, predicted_class_idx].item()

    print(f"Translation successful: {best_word} (confidence: {confidence:.2f})")
    
    return {
        "status": "success", 
        "gloss": best_word, 
        "confidence": confidence
    }