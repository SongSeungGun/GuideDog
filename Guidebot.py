import sys
import cv2
import torch
import time
from PySide6.QtWidgets import QMainWindow, QApplication
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QTimer, QMetaObject, Qt
from threading import Thread
from gtts import gTTS
import pygame
import os

from Guide_ui import Ui_MainWindow

cls_names = ['Bicycle', 'Car', 'Chair', 'Crosswalk', 'Desk', 'Door', 'Electric-Kickboard', 'Elevator', 'Escalator', 'Human', 'Motorcycle', 'Stairs', 'Traffic light-green-', 'Traffic light-red-', 'Tree', 'Utility Pole', 'Wall', 'pole']

class MainWindow(QMainWindow, Ui_MainWindow):  # QMainWindow와 Ui_MainWindow 클래스를 상속받아 MainWindow 클래스를 정의합니다.
    def __init__(self):
        super(MainWindow, self).__init__()  # 부모 클래스의 초기화 메서드를 호출하여 초기화합니다.
        self.setupUi(self)  # UI 설정 메서드를 호출하여 UI를 초기화합니다.
        # 메인 윈도우 제목 설정
        self.setWindowTitle("Custom Model을 이용한 YOLO 웹캠")  # 메인 윈도우의 제목을 설정합니다.

        # PyTorch 장치 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CUDA가 사용 가능하면 'cuda'를, 그렇지 않으면 'cp0u'를 설정합니다.

        self.model = torch.hub.load("./yolov5",  # YOLOv5 모델이 포함된 로컬 디렉토리 경로
                                    model="custom",  # 사용자 지정 모델을 로드
                                    path="C:/2024_SSG/GuideDog/yolov5/runs/train/Guidebot_model3/weights/best.pt",  # 로드할 모델 가중치 파일의 경로
                                    source="local")  # 로컬 소스에서 모델을 로드

        self.model.to(device=self.device)  # 모델을 self.device (CPU 또는 GPU)로 이동시킵니다.

        self.timer = QTimer()  # 타이머 객체 생성
        self.timer.setInterval(100)  # 100ms마다 타이머 실행 (10fps)

        self.timer.timeout.connect(self.video_pred)  # 매 타이머마다 이 함수를 실행하도록 예약
        self.timer.start()  # 타이머를 시작

        self.video = cv2.VideoCapture(0)  # 기본 웹캠을 열고 비디오 캡처 객체를 생성합니다.
        self.loaded_image = None  # 불러온 이미지를 저장할 변수

        pygame.mixer.init()  # pygame의 mixer 모듈을 초기화하여 사운드 및 음악 재생을 준비합니다.

        Thread(target=self.speak, args=(f"AI 준비됐습니다.",)).start()  # "AI 준비됐습니다."라는 메시지를 말합니다.

        # 프레임 스킵 설정
        self.frame_skip = 5
        self.frame_count = 0
        self.results = None  # Initialize results variable

    def convert2QImage(self, img):  # 배열을 QImage로 변환하여 표시
        height, width, channel = img.shape  # 이미지 배열의 높이, 너비 및 채널 수를 가져옵니다.
        bytes_per_line = width * channel
        return QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)  # QImage 객체로 변환하여 반환합니다.

    def video_pred(self):  # 비디오 감지
        ret, frame = self.video.read()  # 웹캠으로부터 프레임을 읽습니다.
        if not ret:  # 프레임을 읽지 못한 경우
            return
        else:  # 프레임을 성공적으로 읽은 경우
            self.frame_count += 1
            if self.frame_count % self.frame_skip == 0:  # 프레임 스킵 적용
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR 형식의 프레임을 RGB 형식으로 변환합니다.
                start = time.perf_counter()  # 현재 시간(초)을 기록합니다.
                results = self.model(frame)  # 프레임에서 객체 감지를 수행합니다.
                image = results.render()[0]  # 객체 감지 결과를 포함한 이미지를 렌더링하여 가져옵니다.
                self.update_frame(image)  # 변환된 이미지를 QLabel에 표시합니다.
                end = time.perf_counter()  # 현재 시간을 기록합니다.
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR 형식의 프레임을 RGB 형식으로 변환합니다.
                self.update_frame(frame)  # 변환된 프레임을 QLabel에 표시합니다.

            detections = results.xyxy[0].cpu().numpy()  if 'results' in locals() else [] # 결과를 numpy 배열로 변환합니다.
            if len(detections) > 0:  # 탐지된 객체가 있는지 확인합니다.
                sorted_detections = sorted(detections, key=lambda x: x[4], reverse=True)[:1]  # confidence 값 기준으로 정렬 후 상위 1개 선택
                for det in sorted_detections:
                    xyxy = det[:4]  # 바운딩 박스 좌표 (x1, y1, x2, y2)
                    conf = det[4]  # 신뢰도 점수
                    cls = int(det[5])  # 클래스 ID
                    class_name = cls_names[cls] if cls < len(cls_names) else f"클래스 {cls}"  # 클래스 이름 또는 ID
                    print(f'xyxy={xyxy}')  # 바운딩 박스 좌표 출력
                     

                if class_name == 'Human':  # 감지된 객체가 'person'인 경우
                    Thread(target=self.speak, args=(f"전방에 사람이 보입니다.",)).start()  # 음성 출력

                if class_name == 'traffic_light':  # 감지된 객체가 'traffic_light'인 경우
                    Thread(target=self.speak, args=(f"전방에 신호등이 보입니다. 신호 판독을 시작하겠습니다.",)).start()  # 음성 출력

                if class_name == 'Traffic light-green-':  # 감지된 객체가 'Traffic light-green-'인 경우
                    Thread(target=self.speak, args=(f"판독결과 신호가 보행신호입니다. 건너도 좋습니다.",)).start()  # 음성 출력

                if class_name == 'Traffic light-red-':  # 감지된 객체가 'Traffic light-red-'인 경우
                    Thread(target=self.speak, args=(f"판독결과 신호가 정지신호입니다. 보행하면 안됩니다.",)).start()  # 음성 출력

                if class_name == 'Wall':  # 감지된 객체가 'wall'인 경우
                    Thread(target=self.speak, args=(f"가까운곳에 벽이 있습니다. 부딪히지 않게 조심하세요.",)).start()  # 음성 출력

                if class_name == 'Car':  # 감지된 객체가 'wall'인 경우
                    Thread(target=self.speak, args=(f"전방에 자동차가 보입니다.위험하니 부딪히지 않게 천천히 보행하세요.",)).start()  # 음성 출력

                if class_name == 'Tree':  # 감지된 객체가 'Tree'인 경우
                    Thread(target=self.speak, args=(f"앞에 나무가 보여요",)).start()  # 음성 출력













    def closeEvent(self, event):  # 윈도우가 닫힐 때 호출되는 이벤트 핸들러
        event.accept()  # 종료 이벤트를 수락하여 애플리케이션이 정상적으로 종료되도록 함
        
    def speak(self, text=None, mp3_path=None):  # text 또는 mp3_path 중 하나를 인수로 받습니다.
        
        if mp3_path:  # mp3_path가 주어진 경우
            pygame.mixer.music.load(mp3_path)  # 주어진 MP3 파일을 로드합니다.
            pygame.mixer.music.play()  # MP3 파일을 재생합니다.
            while pygame.mixer.music.get_busy():  # 재생 중일 때 대기합니다.
                pygame.time.Clock().tick(10)  # 10ms마다 대기

        elif text:  # text가 주어진 경우
            tts = gTTS(text=text, lang='ko')  # 주어진 텍스트를 음성으로 변환합니다.
            tts.save("Javis.mp3")  # 변환된 음성을 MP3 파일로 저장합니다.
            pygame.mixer.music.load("Javis.mp3")  # 저장된 MP3 파일을 로드합니다.
            pygame.mixer.music.play()  # MP3 파일을 재생합니다.
            while pygame.mixer.music.get_busy():  # 재생 중일 때 대기합니다.
                pygame.time.Clock().tick(10)  # 10ms마다 대기
            pygame.mixer.music.stop()  # MP3 파일 재생 중지
            pygame.mixer.music.unload()  # MP3 파일 언로드
            os.remove("Javis.mp3")  # 파일 삭제

        QMetaObject.invokeMethod(self.timer, "start", Qt.QueuedConnection)  # 타이머를 다시 시작합니다.

    def update_frame(self, img):
        qimage = self.convert2QImage(img)
        pixmap = QPixmap.fromImage(qimage)
        self.video_label.setPixmap(pixmap)
        self.video_label.setScaledContents(True)


if __name__ == "__main__":  # 이 스크립트가 직접 실행될 때만 아래 코드를 실행합니다.
    app = QApplication(sys.argv)  # 애플리케이션 객체를 생성하고, 명령 줄 인수를 전달합니다.
    window = MainWindow()  # MainWindow 객체를 생성합니다.
    window.show()  # MainWindow를 화면에 표시합니다.
    sys.exit(app.exec())  # 애플리케이션의 이벤트 루프를 시작하고, 애플리케이션이 종료될 때 종료 코드를 반환합니다.
