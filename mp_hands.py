import mediapipe as mp
import cv2

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


while True:
    success, frame = cap.read()
    
    frame = cv2.flip(frame, 1)
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            
            mpDraw.draw_landmarks(
                frame, 
                landmarks,
                mpHands.HAND_CONNECTIONS, 
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            x_min = int(min(landmark.x for landmark in landmarks.landmark) * frame.shape[1])
            y_min = int(min(landmark.y for landmark in landmarks.landmark) * frame.shape[0])
            x_max= int(max(landmark.x for landmark in landmarks.landmark) * frame.shape[1])
            y_max = int(max(landmark.y for landmark in landmarks.landmark) * frame.shape[0])

           
            cv2.rectangle(frame, (x_min-40, y_min-30),(x_max+40, y_max+30), (0, 255, 0), 2)
    cv2.imshow("Image", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()