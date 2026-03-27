import cv2
import numpy as np
import os
import mediapipe as mp
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model

# ==========================================
# 1. 全局初始化与辅助函数 (补齐了缺失的核心逻辑)
# ==========================================
mp_holistic = mp.solutions.holistic

def mediapipe_detection(image, model):
    # OpenCV 默认读入是 BGR，MediaPipe 需要 RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                  
    results = model.process(image)                 # 模型预测提取骨骼点
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.RGB2BGR)       
    return image, results

def extract_keypoints(results):
    # 提取 姿态、面部、左手、右手的关键点，如果没有检测到则用全 0 数组占位 (保证维度一致)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


# ==========================================
# 2. 全局加载模型和词汇表 (服务器启动时只加载一次)
# ==========================================
# ⚠️ 注意：这里的词汇表必须跟你训练 h5 模型时的词汇表一模一样！
actions = np.array(['hello', 'thanks', 'iloveyou']) 

# 加载你现成的权重模型
print("⏳ 正在加载 AI 模型，请稍候...")
model = load_model('action.h5') 
print("✅ 模型加载完成！")

# ==========================================
# 3. FastAPI 核心接口
# ==========================================
app = FastAPI()
@app.get("/")
def health_check():
    return {"status": "alive", "message": "Server is running perfectly!"}
@app.get("/ping")
def ping_server():
    print("🚀 [最高优先级] 服务器收到 ping 请求，正在响应！")
    return {"status": "success", "message": "服务器网络和框架完美运行中！"}

@app.post("/translate")
async def translate_video(file: UploadFile = File(...)):
    # 1. 临时保存上传的视频文件
    temp_video_path = f"temp_{file.filename}"
    with open(temp_video_path, "wb") as buffer:
        buffer.write(await file.read())
        
    print(f"✅ [探头1] 视频已存入云端: {temp_video_path}") 

    sequence = []
    cap = cv2.VideoCapture(temp_video_path)
    
    if not cap.isOpened():
        print("❌ [探头2] 糟糕！OpenCV 无法打开这个视频文件！") 
        return {"status": "error", "message": "云端无法解码该视频"}

    # 2. 调用 MediaPipe 进行抽帧和骨骼提取
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("✅ [探头3] 视频画面已全部读取完毕") 
                break 
            
            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            # 只要攒够了 30 帧，立刻停止读取（节省服务器算力和内存）
            if len(sequence) == 30:
                break
                
    cap.release()
    # 3. 极其重要：清理掉临时视频文件，防止服务器硬盘被撑爆
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    
    print(f"📊 [探头4] 最终成功提取到的视频帧数: {len(sequence)}") 

    # 4. 送入模型进行预测
    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        best_word = actions[np.argmax(res)]
        confidence = float(np.max(res)) # 获取最高置信度
        
        print(f"🎉 [探头5] 预测成功: {best_word} (置信度: {confidence:.2f})")
        
        return {
            "status": "success", 
            "gloss": best_word,
            "confidence": confidence
        }
    else:
        print("⚠️ [探头6] 帧数不足 30 帧，无法进行 AI 预测！")
        return {
            "status": "error", 
            "message": f"视频太短或画面无法识别，只提取到了 {len(sequence)} 帧（需要 30 帧）"
        }