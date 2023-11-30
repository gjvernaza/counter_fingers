import mediapipe as mp
import cv2
import time


path = "counter_fingers/data/C"
count = 0

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


while True:
    success, frame = cap.read()

    frame = cv2.flip(frame, 1)
    frame_output = frame.copy()
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
           
            x_min = int(
                min(landmark.x for landmark in landmarks.landmark) * frame.shape[1])
            y_min = int(
                min(landmark.y for landmark in landmarks.landmark) * frame.shape[0])
            x_max = int(
                max(landmark.x for landmark in landmarks.landmark) * frame.shape[1])
            y_max = int(
                max(landmark.y for landmark in landmarks.landmark) * frame.shape[0])
            x_min = x_min - 40
            y_min = y_min - 40
            x_max = x_max + 40
            y_max = y_max + 40
            bbox = x_min, y_min, x_max, y_max
            cv2.rectangle(frame_output, (x_min, y_min),
                          (x_max, y_max), (0, 255, 0), 2)
            box_width = x_max - x_min
            box_height = y_max - y_min

            zone_detection = frame[y_min:y_max, x_min:x_max]
           
            if zone_detection is not None and zone_detection.shape[0] > 0 and zone_detection.shape[1] > 0:
                zone_resized = cv2.resize(zone_detection, (224, 224))
                image = cv2.resize(zone_resized, (224, 224),
                                   interpolation=cv2.INTER_AREA)
                

                cv2.imshow("Zone", zone_detection)
                if image is not None and image.shape[0] > 0 and image.shape[1] > 0:
                    cv2.imshow("Zone Resized", zone_resized)
                # cv2.imshow("Zone White", imgWhite)
                
    cv2.imshow("Image", frame_output)
    key = cv2.waitKey(1)
    if key == ord('q') or count==300 :
        print("Saved")
        break
    if key == ord('s'):
        count+=1
        cv2.imwrite(f"{path}/image_{time.time()}.jpg", image)
        
cap.release()
cv2.destroyAllWindows()
