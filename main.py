from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model

app = FastAPI()

# ==========================================
# 1. 照抄安全区代码 (直接从 Notebook 复制)
# ==========================================
mp_holistic = mp.solutions.holistic
# (把 Notebook 里的 mediapipe_detection 和 extract_keypoints 函数完整复制到这里)
def extract_keypoints(results):
    # ... 原封不动 ...
    return np.concatenate([pose, face, lh, rh])

# ==========================================
# 2. 全局加载现成的模型和词汇表
# ==========================================
actions = np.array(['hello', 'thanks', 'iloveyou'])
model = load_model('action.h5') # 直接加载他现成的权重！

# ==========================================
# 3. 核心 API 接口
# ==========================================
@app.post("/translate")
async def translate_video(file: UploadFile = File(...)):
    temp_video_path = f"temp_{file.filename}"
    with open(temp_video_path, "wb") as buffer:
        buffer.write(await file.read())
        
    print(f"✅ [探头1] 视频已存入云端: {temp_video_path}") 

    sequence = []
    cap = cv2.VideoCapture(temp_video_path)
    
    if not cap.isOpened():
        print("❌ [探头2] 糟糕！OpenCV 无法打开这个视频文件！") 
        return {"status": "error", "message": "云端无法解码该视频"}

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("✅ [探头3] 视频画面已全部读取完毕") 
                break 
            
            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            if len(sequence) == 30:
                break
                
    cap.release()
    os.remove(temp_video_path)
    
    print(f"📊 [探头4] 最终成功提取到的视频帧数: {len(sequence)}") 

    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        best_word = actions[np.argmax(res)]
        print(f"🎉 [探头5] 预测成功，结果是: {best_word}")
        return {"status": "success", "gloss": best_word}
    else:
        # 这里就是你之前可能漏掉的兜底 return
        print("⚠️ [探头6] 帧数不足 30 帧，无法进行 AI 预测！")
        return {"status": "error", "message": f"视频太短或画面无法识别，只提取到了 {len(sequence)} 帧"}