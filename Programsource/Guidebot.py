import sys
import cv2
import torch
import time
from PySide6.QtWidgets import QMainWindow, QApplication
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QTimer, QMetaObject, Qt
from threading import Thread, Lock, Event
import pyttsx3
import numpy as np
from queue import Queue
from Guide_ui import Ui_MainWindow
from cam import Depth  
import openai
import asyncio
import concurrent.futures
import speech_recognition as sr

cls_names = ['자전거', '자동차', '의자', '횡단보도', '책상', '문', '킥보드', '엘리베이터', '에스컬레이터', 
             '사람', '오토바이', '계단', '녹색 신호', '적색 신호', '나무', '전봇대', '벽', '기둥']

class OpenAi:
    def __init__(self, api_key):
        openai.api_key = api_key 
        self.engine = pyttsx3.init()  
        self.content = '' 
    
    def text_to_voice(self, text):
        self.engine.say(text) 
        self.engine.runAndWait()
  
    def voice_to_text(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print('say something:')
            audio = r.listen(source)
            said = ""

            try:
                said = r.recognize_google(audio, language="ko-KR")
                print("나: ", said)
            except sr.UnknownValueError:
                self.text_to_voice("죄송합니다 다시 한번 말씀해주시겠어요? \n")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")

        return said

    def fetch_gpt_response(self, messages):
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
        return response['choices'][0]['message']['content'].strip()

    async def get_gpt_response(self, messages):
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            response = await loop.run_in_executor(pool, self.fetch_gpt_response, messages)
        return response

    async def chat_loop(self):
        self.text_to_voice("안녕하세요 대화를 시작합니다. \n")
        self.text_to_voice("대화 종료를 원하시면 '종료'라고 말하세요. \n")

        while True:
            prompt = self.voice_to_text()

            if '종료' in prompt:
                self.text_to_voice("대화를 종료합니다.")
                break

            try:
                messages = [
                    {'role': 'system', 'content': 'You are a helpful assistant who responds in Korean.'},
                    {'role': 'user', 'content': self.content},
                ]
                messages.append({'role': 'user', 'content': prompt})
            except Exception as e:
                print(f"An error occurred: {e}")
                messages = [
                    {'role': 'system', 'content': 'You are a helpful assistant who responds in Korean.'},
                    {'role': 'user', 'content': prompt},
                ]

            response = await self.get_gpt_response(messages)

            print("------------------------------")
            print(response)
            print("------------------------------")

            self.text_to_voice(response)

            await asyncio.sleep(0.1)

            self.content += response

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("시각장애를 위한 객체인식 프로그램")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("ultralytics/yolov5", model="custom", path="./best.pt")
        self.model.to(device=self.device)

        self.depth_camera = Depth()
        self.depth_camera.start_pipeline()

        self.speech_queue = Queue()
        self.speech_lock = Lock()
        self.speech_event = Event()

        self.speech_thread = Thread(target=self.process_speech_queue)
        self.speech_thread.daemon = True
        self.speech_thread.start()

        self.speak("객체인식 프로그램을 시작합니다.")

        self.processing_speech = False

        self.frame_skip = 2
        self.frame_count = 0
        self.results = None

        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start()

        self.openai_api_key = ''
        self.ai = OpenAi(api_key=self.openai_api_key)

        self.chat_thread = Thread(target=self.run_chat_loop)
        self.chat_thread.daemon = True
        self.chat_thread.start()

    def run_chat_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.ai.chat_loop())

    def convert2QImage(self, img):
        height, width, channel = img.shape
        bytes_per_line = width * channel
        return QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)

    def video_pred(self):
        depth_image, color_image = self.depth_camera.get_frame()
        if color_image is None:
            return
        
        else:
            self.frame_count += 1
            if self.frame_count % self.frame_skip == 0:
                frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 640))
                start = time.perf_counter()
                results = self.model(frame)
                image = results.render()[0]
                self.update_frame_ui(self.label, image)
                end = time.perf_counter()
                detections = results.xyxy[0].cpu().numpy()
                if len(detections) > 0:
                    sorted_detections = sorted(detections, key=lambda x: x[4], reverse=True)[:1]
                    for det in sorted_detections:
                        x1, y1, x2, y2, conf, cls = det
                        cls = int(cls)
                        class_name = cls_names[cls] if cls < len(cls_names) else f"클래스 {cls}"

                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)

                        if 0 <= center_x < depth_image.shape[1] and 0 <= center_y < depth_image.shape[0]:
                            distance = depth_image[center_y, center_x] * 0.001
                        else:
                            return

                        distance_rounded = round(distance, 2)

                        particle = "이" if class_name.endswith(('상', '문', '이', '색', '람', '단', '벽', '둥')) else "가"
                        message = f"{class_name}{particle} {distance_rounded} 미터 앞에 있습니다."

                        self.speak(message)

    def update_frame(self):
        depth_image, color_image = self.depth_camera.get_frame()
        if color_image is None:
            return

        frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        self.update_frame_ui(self.label, frame)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        self.update_frame_ui(self.label_2, depth_colormap)

        if not self.processing_speech:
            self.video_pred()

    def process_speech_queue(self):
        engine = pyttsx3.init()
        while True:
            message = self.speech_queue.get()
            if message is None:
                break
            with self.speech_lock:
                self.processing_speech = True
                print(f"Speaking: {message}")
                engine.say(message)
                engine.runAndWait()
                self.processing_speech = False

    def speak(self, text):
        self.speech_queue.put(text)

    def closeEvent(self, event):
        self.depth_camera.stop_pipeline()
        self.speech_queue.put(None)
        event.accept()

    def update_frame_ui(self, label, img):
        qimage = self.convert2QImage(img)
        pixmap = QPixmap.fromImage(qimage)
        label.setPixmap(pixmap)
        label.setScaledContents(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
