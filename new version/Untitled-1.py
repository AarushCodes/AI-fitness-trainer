# %%
import cv2
import numpy as np
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
import speech_recognition as sr


# %%
def normalize_landmarks(landmarks):
    x_values = []
    y_values = []
    for landmark in landmarks:
    #if landmark.visibility >= 0.50:
        x_values.append(landmark.x)
        y_values.append(landmark.y)
    
    min_x = min(x_values)
    min_y = min(y_values)
    normalized_x  = [x - min_x for x in x_values]

    normalized_y = [y- min_y for y in y_values]
    #normalized_landmarks = list(zip(normalized_x, normalized_y))
    normalized_landmarks = [item for pair in zip(normalized_x, normalized_y) for item in pair]
    #print(normalized_landmarks)
    return normalized_landmarks
    


# %%
def isnarroworwide(feetlandmarks):
    #feetlandmarks  = feetlandmarks[-4:]
    legonedifference = feetlandmarks[1] - feetlandmarks[3]
    legtwodifference = feetlandmarks[2] - feetlandmarks[0]
    #print(legonedifference, ' ', legtwodifference)
    if (legonedifference > -0.01 and legonedifference < 0.01) or (legtwodifference > -0.01 and legtwodifference < 0.01) :
        status = 'perfect'
    elif legonedifference > 0.01 or legtwodifference > 0.01:
        status = 'wide'
    elif legonedifference < -0.01 or legtwodifference < -0.01:
        status = 'narrow'

    else:
        status = 'NA'
    return status
    

# %%
def are_legs_too_open_or_closed(landmarks):
    diff1 = landmarks[30] - landmarks[12]
    diff2 = landmarks[11] -  landmarks[29]
    if (diff1 > -0.015 and diff1 < 0.015) or (diff2 > -0.015 and diff2 < 0.015):
        status = 'perfect'
    elif diff1< -0.015 or diff2< -0.015:
        status = 'too open'
    elif diff1 > 0.015 or diff2< 0.015:
        status = 'too closed'
    else:
        status = 'NA'
    
    return status

# %%
class checkExcercise:

    def __init__(self, state):
        self.state = state

    def calculateangle(self,a,b,c): 
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians*180/np.pi)
        #print(angle)
        if angle > 180:
            angle = 360 - angle
        return angle
    

    def isBicepCurl(self,results,mode='left'):
        #print('mode is ', mode)
        if mode == 'Left':
            #print(mode)
            landmarks = results.pose_landmarks.landmark
            lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            lelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            lwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            angle = self.calculateangle(lshoulder,lelbow,lwrist)
            
            if angle > 160:
                #print('down')
                self.state = 'down'
                #print('gottohere')
            if angle < 30 and self.state =='down':
                #print('up')
                self.state = 'up'
                #print('gottohere')
                return True
                #print(counter)
        elif mode == 'Right':
            landmarks = results.pose_landmarks.landmark
            rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            angle = self.calculateangle(rshoulder,relbow,rwrist)
            if angle > 160:
                #print('down')
                self.state = 'down'
            if angle < 30 and self.state =='down':
                #print('up')
                self.state = 'up'
                return True
        elif mode == 'Both':
            landmarks = results.pose_landmarks.landmark
            rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            lelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            lwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            rangle = self.calculateangle(rshoulder,relbow,rwrist)
            langle = self.calculateangle(lshoulder,lelbow,lwrist)
            if rangle > 160 and langle > 160:
                #print('down')
                self.state = 'down'
            if rangle < 30 and langle < 30 and self.state =='down':
                #print('up')
                self.state = 'up'
                return True
            

# %%
import queue
def recognize_speech(recognizer,result_queue,microphone):
    while True:
        with microphone as source:
            print("Say something:")
            audio = recognizer.listen(source)
        try:
            recognized_text = recognizer.recognize_google(audio)
            result_queue.put(recognized_text)
        except sr.UnknownValueError:
            #recognized_text = None
            pass
        except sr.RequestError as e:
            #recognized_text = None
            pass
        #return recognized_text


# %%
recognizer = sr.Recognizer()



# %%
import threading
#r = sr.Recognizer()
recognizer = sr.Recognizer()
#model = keras.models.load_model('model2.h5')
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(['up', 'down'])
selection = -1
counter = 0
selection_speed = 7
squatorbicepcentres = [(1038, 273), (1038, 559)]
selectbicepcentres = [(798+150,214), (798+330, 395), (798+150,576)]
backgroundimg = cv2.imread('D:/python projects/ai cv/gym traniner/NEW edition/backgroundimage.png')
# mode1 = cv2.imread('./modes/1.png')
# perfectmode = cv2.imread('perfect.png')
# wrongmode = cv2.imread('wrongv2.png')
# selectbicepcurlmode = cv2.imread('selectbicepcurl.png')
# bicepmode = cv2.imread('bicep.png')
#checkExcercise = checkExcercise(state='None')
state = None
reps = 0
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 3
font_color = (255, 255, 255)
checkhands = True
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) 
mode = 'start'
cap = cv2.VideoCapture(0)
#detector = HandDetector(detectionCon=0.8, maxHands=1)
microphone = sr.Microphone()
result_queue = queue.Queue()
recognition_thread = threading.Thread(target=recognize_speech, args=(recognizer, result_queue,microphone))
recognition_thread.daemon = True  # Daemonize the thread so it doesn't block program exit
recognition_thread.start()
while True:
    ret, image = cap.read()
    imgbackground = backgroundimg
    imgbackground[171:171+480, 69:69+640] = image
    if not result_queue.empty():
        print('sewyy')
        recognized_text = result_queue.get()
        print(recognized_text)
        cv2.putText(imgbackground, str(recognized_text), (738+250,200), font, 3, font_color, 5, cv2.LINE_AA)
    



    cv2.imshow('trainer', backgroundimg)
    
    #print('yeah boy')
    if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

# %% [markdown]
# 

# %%


# %%
model = keras.models.load_model('model2.h5')
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(['up', 'down'])
selection = -1
counter = 0
selection_speed = 7
squatorbicepcentres = [(1038, 273), (1038, 559)]
selectbicepcentres = [(798+150,214), (798+330, 395), (798+150,576)]
imgbackground = cv2.imread('backgroundimage.png')
mode1 = cv2.imread('./modes/1.png')
perfectmode = cv2.imread('perfect.png')
wrongmode = cv2.imread('wrongv2.png')
selectbicepcurlmode = cv2.imread('selectbicepcurl.png')
bicepmode = cv2.imread('bicep.png')
checkExcercise = checkExcercise(state='None')
state = None
reps = 0
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 3
font_color = (255, 255, 255)
checkhands = True
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) 
mode = 'select squat and bicep'
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)
while cap.isOpened():
    ret, image = cap.read()
    if checkhands:
        hands, image = detector.findHands(image)
        imgbackground[171:171+480, 69:69+640] = image
        if hands:
            hand1 = hands[0]
            fingers1 = detector.fingersUp(hand1)
    
    
    
    if mode =='select squat and bicep' and hands:
        imgbackground[0:720, 798:1280] = mode1

        if fingers1 == [0,1,0,0,0]:
            if selection != 'squat':
                counter = 1
            selection = 'squat'
            centre = squatorbicepcentres[0]
        elif fingers1 == [0,1,1,0,0]:
            if selection != 'bicep':
                counter = 1
            selection = 'bicep'
            centre = squatorbicepcentres[1]
        else:
            selection = -1
            counter = 0

        if counter > 0 :
            counter += 1
            #print(counter)
            cv2.ellipse(imgbackground,centre, (105,105), 0, 0,counter*selection_speed, (255,255,255),20)
            #print(counter)
            if counter*selection_speed > 360:
                selected = selection
                print(selected)
                counter = 0
                
                if selected == 'bicep':
                    mode = 'select bicep mode'
                    print(mode)
                else:
                    mode = 'squats'
                    checkhands = False
        cv2.imshow('Your gym buddy', imgbackground)
    elif mode == 'select squat and bicep':
        imgbackground[171:171+480, 69:69+640] = image
        imgbackground[0:720, 798:1280] = mode1
        cv2.imshow('Your gym buddy', imgbackground)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    elif mode == 'squats':
        #with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #image = cv2.resize(image, 480,640)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            imgbackground[171:171+480, 69:69+640] = image
            try:
                landmarks = results.pose_landmarks.landmark
                
                normalized_landmarks = normalize_landmarks(landmarks)
                testdata = np.array([normalized_landmarks])
                predictions = model.predict(testdata)
                predicted_classes = (predictions > 0.5).astype(int)

                #predicted_classes = np.argmax(predictions, axis=1)
                upordown = label_encoder.inverse_transform(predicted_classes.flatten()).tolist()[0]
                landmarks = [landmark.x for landmark in landmarks]
                #print(f'Predicted Class: {predicted_classes_labels}')  
                #normalized_landmarks = normalize_landmarks(landmarks)
                #print(isnarroworwide(landmarks[-4:]))
                narroworwide = isnarroworwide(landmarks[-4:])
                too_open_or_closed = are_legs_too_open_or_closed(landmarks)

                if narroworwide == 'perfect' and too_open_or_closed == 'perfect':
                     imgbackground[0:720, 798:1280] = perfectmode
                     if upordown == 'down':
                          state = 'down'
                     if upordown == 'up' and state =='down':
                          state = 'up'
                          reps += 1
                else:
                    imgbackground[0:720, 798:1280] = wrongmode
                    y_position = 500
                    
                    wronglines = []
                    if narroworwide == 'wide':
                        wronglines.append('Your toes are facing outwards')
                    elif narroworwide == 'narrow':
                        wronglines.append('Your toes are facing inwards')
                    
                    if too_open_or_closed == 'too open':
                        wronglines.append('Your legs are too open')
                    elif too_open_or_closed == 'too closed':
                        wronglines.append('Your legs are too closed')
                    #text_lines = [f'Toes status: {narroworwide}', f'Legs width status: {too_open_or_closed}', f'Number of reps: {counter}', f'Position: {upordown}']
                    for line in wronglines:
                        cv2.putText(imgbackground, line, (738+80, y_position), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                        y_position += 60
                cv2.putText(imgbackground, str(reps), (738+250,200), font, 3, font_color, 5, cv2.LINE_AA)
                # diff1 = landmarks[30] - landmarks[12]
                # diff2 = landmarks[11] -  landmarks[29] 
                #print(diff1, 'second: ', diff2)
                # cv2.putText(image, f'''second:{diff2}''',(50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                #cv2.putText(image, f'norw: {narroworwide}, oc: {too_open_or_closed}, pos:{predicted_classes_labels}',(50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                #print(image.shape)
                #print(topandbottom(landmarks))
                #topandbottom(landmarks,resolution)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                        ) 
            except Exception as e:
                print(e)
                #pass
            
            #print(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value])
            
            
            imgbackground[171:171+480, 69:69+640] = image
            cv2.imshow('Your gym buddy', imgbackground)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    elif mode == 'select bicep mode':
        #print('in select bicep mode')
        imgbackground[0:720, 798:1280] = selectbicepcurlmode
        if fingers1 == [0,1,0,0,0]:
            if selection != 'Left':
                counter = 1
            selection = 'Left'
            centre = selectbicepcentres[0]
        elif fingers1 == [0,1,1,0,0]:
            if selection != 'Right':
                counter = 1
            selection = 'Right'
            centre = selectbicepcentres[1]
        elif fingers1 == [0,1,1,1,0]:
            if selection != 'Both':
                counter = 1
            selection = 'Both'
            centre = selectbicepcentres[2]
        else:
            counter = 0
        if counter >0:
            counter += 1
            cv2.ellipse(imgbackground,centre, (90,90), 0, 0,counter*selection_speed, (255,255,255),20)
            if counter*selection_speed >= 360:
                mode = f'{selection} bicep'
                checkhands = False
                print(mode.removesuffix(' bicep'))
        cv2.imshow('Your gym buddy', imgbackground)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    elif 'bicep' in mode:
        imgbackground[0:720, 798:1280] = bicepmode
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
                # Make detection
            results = pose.process(image)
            
                # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            isbicepcurl = checkExcercise.isBicepCurl(results,mode=mode.removesuffix(' bicep'))
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                        ) 
            if isbicepcurl:
                reps += 1
            imgbackground[171:171+480, 69:69+640] = image
            
        except:
            pass
        
        cv2.putText(imgbackground, str(reps), (738+250,200), font, 3, font_color, 5, cv2.LINE_AA)
        cv2.imshow('Your gym buddy', imgbackground)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


# %%
mode

# %%


# %%
(377 + 169) / 2

# %%
(136+345) / 2

# %%
240.5 + 798

# %%



