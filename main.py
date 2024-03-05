from ear_detect import *

import cv2
import mediapipe as mp
import time
from imageai.Detection import ObjectDetection

vid = VideoHandler()
cap = cv2.VideoCapture('./armenia.MOV')

fps = 30
width = 1280
height = 720
cap.set(cv2.CAP_PROP_FPS, fps)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

th = {"EAR_THRESH": 0.2}
mp_holistic = mp.solutions.holistic
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath("./model/retinanet_resnet50_fpn_coco-eeacb38b.pth")
detector.loadModel()
output_path = './out_image.jpg'

array_detection = []
count = 0
def phone_obj(obje,list=[]):
    global count
    print(count)
    if obje == "cell phone":
        start_phone = time.time()
        list.append(start_phone)
        print(start_phone)
        count += 1
        if count >= 4:
            finish_phone  = time.time()
            print(finish_phone-list[0])

            print(f"{int(finish_phone-list[0])}The person is not interested, he is on the phone")
finish = 0

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic: 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        start = time.time()
        if start - finish > 2:
            detections = detector.detectObjectsFromImage(input_image=frame, output_image_path=output_path,
                                                         minimum_percentage_probability=10)
            finish = time.time()
            for i in detections:
                print(i["name"])
                phone_obj(i["name"])
        

        processed_frame = vid.process_video(frame, th)
        cv2.imshow('EAR detect', processed_frame)

        if cv2.waitKey(1) &  0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
