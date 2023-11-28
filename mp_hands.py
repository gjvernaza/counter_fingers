import mediapipe as mp
import cv2
import numpy as np
from keras.models import load_model


np.set_printoptions(suppress=True)
model = load_model("counter_fingers/model/model_keras_100_epochs/keras_Model.h5", compile=False)
class_names = open("counter_fingers/model/model_keras_100_epochs/labels.txt", "r").readlines()

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
            x_min = x_min - 40
            y_min = y_min - 40
            x_max = x_max + 40
            y_max = y_max + 40
            bbox = x_min, y_min, x_max, y_max
            cv2.rectangle(frame_output, (x_min, y_min),
                          (x_max, y_max), (0, 255, 0), 2)
            box_width = x_max - x_min
            box_height = y_max - y_min

            #imgWhite = np.ones((300, 300, 3), np.uint8) * 255

            zone_detection = frame[y_min:y_max, x_min:x_max]
            
            
                
            #print(zone_detection.shape)
            if zone_detection is not None and zone_detection.shape[0] > 0 and zone_detection.shape[1] > 0:
                zone_resized = cv2.resize(zone_detection, (224, 224))
                image = cv2.resize(zone_resized, (224, 224),
                                   interpolation=cv2.INTER_AREA)
                image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

                # Normalize the image array
                image = (image / 127.5) - 1

                # Predicts the model
                prediction = model.predict(image)
                index = np.argmax(prediction)
                class_name = class_names[index]
                confidence_score = prediction[0][index]
                print("Class:", class_name, end="")
                print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
                
                
                class_name = class_name[2:3]
                classes = ["A", "B", "C"]

                if class_name == "A" and confidence_score>0.6:
                    class_name = classes[0]
                
                elif class_name == "B" and confidence_score>0.8:
                    class_name = classes[1]
                elif class_name == "C" and confidence_score>0.6:
                    class_name = classes[2]
                else:
                    class_name = ""
                    confidence_score = 0.0
                    

                # Print prediction and confidence score
                
                cv2.imshow("Zone", zone_detection)
                if image is not None and image.shape[0] > 0 and image.shape[1] > 0:
                    cv2.imshow("Zone Resized", zone_resized)
                #cv2.imshow("Zone White", imgWhite)
                cv2.putText(frame_output, class_name, (10, 50),
                            cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2),
    cv2.imshow("Image", frame_output)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()