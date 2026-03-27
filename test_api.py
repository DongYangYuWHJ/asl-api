import requests

# 你的真实 Render 云端 API 网址
url = "https://asl-api-iy6e.onrender.com/translate"

# 随便找一个你电脑里的几秒钟的短视频测试（最好小于 2MB，防止免费服务器超时）
# 请把这里的路径换成你电脑里真实存在的 mp4 文件路径
video_path = "test_video.mp4" 

print(f"正在向 {url} 发送视频，请稍候...")

try:
    with open(video_path, "rb") as video_file:
        # 注意：这里的 "file" 必须和你在 main.py 里写的 UploadFile 参数名一模一样
        # 如果你 main.py 里写的是 video: UploadFile，这里就要改成 "video"
        files = {"file": (video_path, video_file, "video/mp4")}
        response = requests.post(url, files=files)

    print("服务器状态码:", response.status_code)
    print("服务器返回内容:", response.text)
    
except Exception as e:
    print("请求出错啦:", e)