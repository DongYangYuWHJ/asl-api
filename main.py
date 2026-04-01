import cv2
import torch
import numpy as np
import os
import imageio
from fastapi import FastAPI, UploadFile, File
from transformers import AutoImageProcessor, AutoModelForVideoClassification

app = FastAPI()

# ==========================================
# 1. 显卡点火与模型加载 (全自动下载)
# ==========================================
# 自动检测你的 4060 Ti，如果没有配置好 CUDA 环境也会自动降级用 CPU 兜底
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 正在启动 AI 引擎，当前运行设备: {device.type.upper()}")

# 设置 Hugging Face 模型 ID
model_ckpt = "Shawon16/VideoMAE_Base_WLASL_100_200_epochs_p20_SR_8"

print("⏳ 正在从 Hugging Face 下载并加载大模型 (初次启动可能需要几分钟，请耐心等待)...")
# 加载图像处理器 (自动负责缩放、归一化) 和 分类模型
# 👇 设定一个你肉眼可见的本地文件夹名字，比如就叫 my_model_weights
local_save_path = "./my_model_weights"

print(f"⏳ 准备下载大模型，文件将安全保存在: {local_save_path} ...")

# 👇 重点在这里：加上 cache_dir=local_save_path 这个参数
image_processor = AutoImageProcessor.from_pretrained(model_ckpt, cache_dir=local_save_path)
model = AutoModelForVideoClassification.from_pretrained(model_ckpt, cache_dir=local_save_path).to(device)
print("✅ 100词汇量 VideoMAE 大模型加载完毕，随时待命！")


# ==========================================
# 2. 视频抽帧工具函数 (VideoMAE 标准需要 16 帧)
# ==========================================
# def read_video(file_path, num_frames=16):
#     cap = cv2.VideoCapture(file_path)
#     frames = []
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     if total_frames < 2:
#         cap.release()
#         return frames

#     # 均匀采样 16 帧
#     indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

#     curr_frame = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if curr_frame in indices:
#             # OpenCV 读进来是 BGR，大模型需要 RGB
#             frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # 注意这里如果是报错可以改回 COLOR_BGR2RGB 测试
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # 修正：标准读取依然是 BGR 转 RGB
#             frames.append(frame)
#         curr_frame += 1
#     cap.release()

#     # 如果视频太短，用最后一帧补齐到 16 帧
#     while len(frames) < num_frames and len(frames) > 0:
#         frames.append(frames[-1])
        
#     return frames


def get_video_frames_fixed(video_path, num_frames=16):
    # imageio 的 ffmpeg 插件会自动读取手机的旋转元数据，直接把画面“扶正”！
    reader = imageio.get_reader(video_path, 'ffmpeg')
    
    # 把它转成列表
    all_frames = [frame for frame in reader]
    total_frames = len(all_frames)
    
    # 均匀抽帧逻辑
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    selected_frames = []
    for idx in indices:
        frame = all_frames[idx]
        # 注意：imageio 读出来默认就是 RGB 格式的！
        # 如果你后面直接交给 transformers 模型，连 cv2.cvtColor(BGR2RGB) 都省了！
        selected_frames.append(frame)
        
    return selected_frames

# ==========================================
# 3. 核心 API 接口
# ==========================================
@app.get("/ping")
def ping_server():
    return {"status": "success", "message": f"4060Ti 重型服务器完美运行中！"}

@app.post("/translate")
async def translate_video(file: UploadFile = File(...)):
    temp_video_path = f"temp_{file.filename}"
    with open(temp_video_path, "wb") as buffer:
        buffer.write(await file.read())
        
    print(f"📥 收到测试视频: {temp_video_path}")

    # 1. 读取并均匀抽取 16 帧画面
    frames = get_video_frames_fixed(temp_video_path, num_frames=16)
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    test_frame = frames[8] # 抽第8帧看看
    cv2.imwrite("./debug_mobile_frame.jpg", test_frame)
    if len(frames) == 0:
         return {"status": "error", "message": "无法读取该视频画面"}

    # 2. 数据预处理并送入 GPU
    inputs = image_processor(list(frames), return_tensors="pt").to(device)

    # 3. 显卡光速推理
    with torch.no_grad(): # 推理模式，不计算梯度以节省显存
        outputs = model(**inputs)
        logits = outputs.logits

    # 4. 解析结果
    predicted_class_idx = logits.argmax(-1).item()
    # 巧妙利用模型自带的词典映射
    best_word = model.config.id2label[predicted_class_idx]
    
    # 计算置信度百分比
    confidence = torch.softmax(logits, dim=-1)[0, predicted_class_idx].item()

    print(f"🎉 翻译成功: {best_word} (置信度: {confidence:.2f})")
    
    return {
        "status": "success", 
        "gloss": best_word, 
        "confidence": confidence
    }