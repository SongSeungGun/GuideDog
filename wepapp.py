import sys
import cv2
import torch
import time
from threading import Thread
from gtts import gTTS
import pygame
import os

from flask import Flask, render_template, Response  # Flask 관련 모듈 임포트

cls_names = ['Bicycle', 'Car', 'Chair', 'Crosswalk', 'Desk', 'Door', 'Electric-Kickboard', 'Elevator', 'Escalator', 'Human', 'Motorcycle', 'Stairs', 'Traffic light-green-', 'Traffic light-red-', 'Tree', 'Utility Pole', 'Wall', 'pole']

app = Flask(__name__)  # Flask 애플리케이션 객체 생성

class VideoCamera:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("./yolov5", model="custom", path="C:/2024_SSG/GuideDog/yolov5/runs/train/Guidebot_model3/weights/best.pt", source="local")
        self.model.to(self.device)
        self.video = cv2.VideoCapture(0)
        pygame.mixer.init()
        Thread(target=self.speak, args=(f"AI 준비됐습니다.",)).start()
        self.frame_skip = 5
        self.frame_count = 0

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None

        self.frame_count += 1
        if self.frame_count % self.frame_skip == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(frame)
            frame = results.render()[0]
            detections = results.xyxy[0].cpu().numpy() if 'results' in locals() else []
            if len(detections) > 0:
                sorted_detections = sorted(detections, key=lambda x: x[4], reverse=True)[:1]
                for det in sorted_detections:
                    cls = int(det[5])
                    class_name = cls_names[cls] if cls < len(cls_names) else f"클래스 {cls}"
                    if class_name == 'Human':
                        Thread(target=self.speak, args=(f"전방에 사람이 보입니다.",)).start()
                    elif class_name == 'traffic_light':
                        Thread(target=self.speak, args=(f"전방에 신호등이 보입니다. 신호 판독을 시작하겠습니다.",)).start()
                    elif class_name == 'Traffic light-green-':
                        Thread(target=self.speak, args=(f"판독결과 신호가 보행신호입니다. 건너도 좋습니다.",)).start()
                    elif class_name == 'Traffic light-red-':
                        Thread(target=self.speak, args=(f"판독결과 신호가 정지신호입니다. 보행하면 안됩니다.",)).start()
                    elif class_name == 'Wall':
                        Thread(target=self.speak, args=(f"가까운곳에 벽이 있습니다. 부딪히지 않게 조심하세요.",)).start()
                    elif class_name == 'Car':
                        Thread(target=self.speak, args=(f"전방에 자동차가 보입니다.위험하니 부딪히지 않게 천천히 보행하세요.",)).start()
                    elif class_name == 'Tree':
                        Thread(target=self.speak, args=(f"앞에 나무가 보여요",)).start()

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def speak(self, text=None, mp3_path=None):
        if mp3_path:
            pygame.mixer.music.load(mp3_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        elif text:
            tts = gTTS(text=text, lang='ko')
            tts.save("Javis.mp3")
            pygame.mixer.music.load("Javis.mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
            os.remove("Javis.mp3")

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    from waitress import serve  # waitress 임포트
    serve(app, host='0.0.0.0', port=8080)  # waitress를 사용하여 애플리케이션 실행
