import cv2
import mediapipe as mp
import copy


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=10, min_detection_confidence=0.5, min_tracking_confidence=0.5)
webcam = cv2.VideoCapture(0)
cap = cv2.VideoCapture("CINEMATIC PORTRAIT VIDEO WITH TERRY _ Fujifilm X-T4.mp4")


def landmark_tracking(source=cap):
    while True:

        ret, frame = source.read()
        if ret is not True:
            break
        height, width, _ = frame.shape
        print("Height, width", height, width)
        RGB_image = copy.deepcopy(frame)

        result = face_mesh.process(RGB_image)
        if result.multi_face_landmarks is not None:
            for facial_landmarks in result.multi_face_landmarks:
                for i in range(0, 468): 
                    pt1 = facial_landmarks.landmark[i]
                    x = int(pt1.x * width)
                    y = int(pt1.y * height)

                    cv2.circle(RGB_image, (x, y), 2, (50, 50, 0), -1)
        cv2.imshow("Video", frame)
        cv2.imshow("Tracking", RGB_image)
        cv2.waitKey(1) 

landmark_tracking(webcam)
