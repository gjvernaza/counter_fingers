from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode=False
)
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

index = [4, 8, 12, 16, 20]

puntos_pulgar = [1, 2, 4]
puntos_palm = [0, 1, 2, 5, 9, 13, 17]
puntos_4_dedos = [8, 12, 16, 20]
puntos_base_4_dedos = [6, 10, 14, 18]

# Colores

green = (48, 255, 0)
blue = (192, 101, 21)
yellow = (0, 204, 255)
purple = (128, 64, 128)
peach = (180, 229, 255)


def centroide(puntos):
    x = 0
    y = 0
    for punto in puntos:
        x += punto[0]
        y += punto[1]
    x = x/len(puntos)
    y = y/len(puntos)
    return [x, y]


app = Flask(__name__)

cap = cv2.VideoCapture(0)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def gen_frames():
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frameRGB)
        thickness = [2, 2, 2, 2, 2]
        # Handedness
        # print("Handedness: ",results.multi_handedness)
        # drawing popints of landmarks
        landmarks = results.multi_hand_landmarks
        if landmarks:
            coordenadas_pulgar = []
            coordenadas_palm = []
            coordenadas_4_dedos = []
            coordenadas_base_4_dedos = []

            for landmark in landmarks:
                for i in puntos_pulgar:
                    x = int(landmark.landmark[i].x * width)
                    y = int(landmark.landmark[i].y * height)
                    coordenadas_pulgar.append([x, y])

                for i in puntos_palm:
                    x = int(landmark.landmark[i].x * width)
                    y = int(landmark.landmark[i].y * height)
                    coordenadas_palm.append([x, y])

                for i in puntos_4_dedos:
                    x = int(landmark.landmark[i].x * width)
                    y = int(landmark.landmark[i].y * height)
                    coordenadas_4_dedos.append([x, y])

                for i in puntos_base_4_dedos:
                    x = int(landmark.landmark[i].x * width)
                    y = int(landmark.landmark[i].y * height)
                    coordenadas_base_4_dedos.append([x, y])

                ####### Puntos del pulgar##########
                p1 = np.array(coordenadas_pulgar[0])
                p2 = np.array(coordenadas_pulgar[1])
                p3 = np.array(coordenadas_pulgar[2])
                ###### Distancias##########
                l1 = np.linalg.norm(p2-p3)
                l2 = np.linalg.norm(p1-p3)
                l3 = np.linalg.norm(p1-p2)
                ###### Angulos##########
                angulo = degrees(acos((l1**2 + l3**2 - l2**2)/(2*l1*l3)))
                thumb_finger = np.array(False)
                if angulo > 150:
                    thumb_finger = np.array(True)
                # print(thumb_finger)

                ###### Centroide########
                cx = int(centroide(coordenadas_palm)[0])
                cy = int(centroide(coordenadas_palm)[1])

                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                coordenadas_centroide = np.array([cx, cy])
                coordenadas_4_dedos = np.array(coordenadas_4_dedos)
                coordenadas_base_4_dedos = np.array(coordenadas_base_4_dedos)
                ######### Distancia######
                dist_centroide_4_dedos = np.linalg.norm(
                    coordenadas_centroide - coordenadas_4_dedos, axis=1)
                dist_centroide_base_4_dedos = np.linalg.norm(
                    coordenadas_centroide - coordenadas_base_4_dedos, axis=1)

                dif = dist_centroide_4_dedos - dist_centroide_base_4_dedos

                # fingers = dif>0

                fingers = np.array([True, True, True, True])

                for j in range(len(dif)):
                    if dif[j] < 0:
                        fingers[j] = False

                fingers = np.append(thumb_finger, fingers)

                for k, f in enumerate(fingers):
                    if f == True:
                        thickness[k] = -1
                # print(fingers)

                ######### Dibujar######
                mp_drawing.draw_landmarks(
                    frame, landmark,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
                finger_count = str(fingers.sum())
                cv2.putText(frame, finger_count, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

        # pulgar
        cv2.rectangle(frame, (100, 10), (150, 60), peach, thickness[0])
        cv2.putText(frame, "Pulgar", (100, 80), 1, 1, peach, 2)
        # indice
        cv2.rectangle(frame, (180, 10), (230, 60), purple, thickness[1])
        cv2.putText(frame, "Indice", (180, 80), 1, 1, purple, 2)
        # medio
        cv2.rectangle(frame, (260, 10), (310, 60), yellow, thickness[2])
        cv2.putText(frame, "Medio", (260, 80), 1, 1, yellow, 2)
        # anular
        cv2.rectangle(frame, (340, 10), (390, 60), green, thickness[3])
        cv2.putText(frame, "Medio", (340, 80), 1, 1, green, 2)
        # meñique
        cv2.rectangle(frame, (420, 10), (470, 60), blue, thickness[4])
        cv2.putText(frame, "Menique", (420, 80), 1, 1, blue, 2)

        flag, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')





if __name__ == '__main__':
    app.run(debug=True)

cap.release()
