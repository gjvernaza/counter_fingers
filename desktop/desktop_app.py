from tkinter import *
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
import cv2
from math import degrees, acos
from functools import partial



#canvas_orange = None
#canvas = None
width_camera = 640
height_camera = 480


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode=False
)

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


def update(cap, canvas, window):
    width = width_camera
    height = height_camera
    thickness = [2, 2, 2, 2, 2]
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)

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

        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        canvas.create_image(0, 0, image=photo, anchor=NW)
        canvas.photo = photo

    window.after(2, lambda: update(cap=cap, canvas=canvas, window=window))


def centroide(puntos):
    x = 0
    y = 0
    for punto in puntos:
        x += punto[0]
        y += punto[1]
    x = x/len(puntos)
    y = y/len(puntos)
    return [x, y]



def main():
    
    global video_enable
    global canvas
    global canvas_orange
    global count
    global cap
    
    video_enable = False
    canvas = None
    canvas_orange = None
    count = 0

    def click_on():
        global video_enable
        global canvas
        global canvas_orange
        global count
        global cap
        video_enable = True
        cap = cv2.VideoCapture(0)

        count+=1
        if count == 1:

        
            if canvas_orange.winfo_exists():
                canvas_orange.destroy()

            canvas = Canvas(
                window,
                width=width_camera,
                height=height_camera,

            )
            canvas.pack(side="right", padx=(0, 50))
            update(cap=cap, window=window, canvas=canvas)

        print(video_enable)
    
    def click_off():
        global video_enable
        global canvas
        global canvas_orange
        global count
        count = 0
        video_enable = False
        if canvas.winfo_exists():
            canvas.destroy()
            cap.release()

        canvas_orange = Canvas(
            window,
            width=width_camera,
            height=height_camera,
            background="orange"

        )
        canvas_orange.pack(side="right", padx=(0, 50))
        print(video_enable)

    
    window = Tk()
    window.geometry("1000x600")
    window.resizable(False, False)
    window.title("Contador de dedos con Visión Artificial")

    

    lbl = Label(
        window,
        text="Contador de dedos con Visión Artificial",
        font=("Arial Bold", 24)
    )
    lbl.pack(side=TOP, fill="none")

    button_on = Button(
        window,
        text="Iniciar",
        default="disabled",
        width=15,
        pady=10,
        background="blue",
        #command=partial(on_click_on,window=window)
        #command=lambda: on_click_on(window=window)
        command=click_on
    )
    button_on.pack(side="left", fill="x", padx=(40, 20))

    button_off = Button(
        window,
        text="Detener",
        default="disabled",
        width=15,
        pady=10,
        background="red",
        #command=partial(on_click_off,window=window, cap=cap)
        #command=lambda: on_click_off(window=window, cap=cap)
        command=click_off
    )

    button_off.pack(side="left", fill="x")

    if not video_enable:
        
        canvas_orange = Canvas(
            window,
            width=width_camera,
            height=height_camera,
            background="orange"

        )
        canvas_orange.pack(side="right", padx=(0, 50))
        
    
        
    window.mainloop()
    
    
    if cap.isOpened():
        cap.release()


if __name__ == "__main__":
    main()
