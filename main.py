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
    # a. 把手机传来的视频存到服务器本地临时文件
    temp_video_path = f"temp_{file.filename}"
    with open(temp_video_path, "wb") as buffer:
        buffer.write(await file.read())

    # b. 核心推断逻辑：读取视频 -> 抽帧 -> 进模型
    sequence = []
    cap = cv2.VideoCapture(temp_video_path)
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # 视频读完了
            
            # 把帧喂给 MediaPipe
            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            # 为了匹配作者的模型，只取前 30 帧（或者根据实际情况处理长度）
            if len(sequence) == 30:
                break
                
    cap.release()
    os.remove(temp_video_path) # 阅后即焚，删掉临时视频

    # c. 预测并返回结果
    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        best_word = actions[np.argmax(res)]
        return {"status": "success", "gloss": best_word}
    else:
        return {"status": "error", "message": "视频太短或无法识别"}