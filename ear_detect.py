import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates
import time
from imageai.Detection import ObjectDetection #pip install cython pillow>=7.0.0 numpy>=1.18.1 opencv-python>=4.1.2 torch>=1.9.0 --extra-index-url https://download.pytorch.org/whl/cpu torchvision>=0.10.0 --extra-index-url https://download.pytorch.org/whl/cpu pytest==7.1.3 tqdm==4.64.1 scipy>=1.7.3 matplotlib>=3.4.3 mock==4.0.3


def get_mp_app(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
):
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    ) # инициализация фейс меш

    return face_mesh

def distance(p1, p2): # расстояние между двумя точками
    return sum([(i - j) ** 2 for i, j in zip(p1, p2)]) ** 0.5

def get_ear(landmarks, idxs, frame_width, frame_height): # вычисление ЕАР для глаза
    try:
        coords_points = []
        for i in idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
            coords_points.append(coord)

        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])

        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)

    except:
        ear = 0.0
        coords_points = None

    return ear, coords_points

def get_average_ear(landmarks, l_eye_indxs, r_eye_indxs, i_w, i_h): # средний ЕАР
    left_ear, left_lm_coordinates = get_ear(landmarks, l_eye_indxs, i_w, i_h)
    right_ear, right_lm_coordinates = get_ear(landmarks, r_eye_indxs, i_w, i_h)
    Avg_EAR = (left_ear + right_ear) / 2.0
    return Avg_EAR, (left_lm_coordinates, right_lm_coordinates)



def plot_text(image, text, origin, color, font=cv2.FONT_HERSHEY_SIMPLEX, fntScale=0.8, thickness=2):
    # Убедитесь, что изображение доступно для записи
    image.flags.writeable = True
    image = cv2.putText(image, text, origin, font, fntScale, color, thickness)
    return image

#Обработка видео по кадрам
class VideoHandler: 
    
    def __init__(self):
        self.indxs = {
            'left': [362, 385, 387, 263, 373, 380],
            'right': [33, 160, 158, 133, 153, 144],
        }
        self.RED = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.facemesh_model = get_mp_app()
        self.EAR_txt_pos = (10, 30)
        self.static = {
            "COLOR": self.GREEN,
        }
        self.EAR_txt_pos = (10, 30)
    
    def process_video(self, frame: np.array, thresholds: dict):
        frame.flags.writeable = False
        frame_h, frame_w, _ = (1280, 720, 4)

        results = self.facemesh_model.process(frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            EAR, coordinates = get_average_ear(landmarks, self.indxs["left"], self.indxs["right"], frame_w, frame_h)
            

            if EAR < thresholds["EAR_THRESH"]:
                plot_text(frame, 'eyes closed', (50, 50), self.static["COLOR"])
                print("Глаза закрыты")
            else:
                self.static['COLOR'] = self.GREEN
            
            EAR_txt = f'EAR: {round(EAR, 2)}'
            plot_text(frame, EAR_txt, self.EAR_txt_pos, self.static["COLOR"])
        else:
            self.static["COLOR"] = self.GREEN
            frame = cv2.flip(frame, 1)
        return frame
    