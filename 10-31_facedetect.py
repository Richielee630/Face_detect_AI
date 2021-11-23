import pickle
import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from skimage.feature import hog

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

gender_model_name = 'svc_model_v1.1.h5'
gender_pca_name = 'pca_90_eivector.pca'

age_model_name ='svc_age_model_v1.1.h5'
age_pca_name ='pca_90_age_eivector_v1.1.pca'

with open(gender_model_name, 'rb') as gender_model:
    SVC_gender_model = pickle.load(gender_model)

with open(gender_pca_name, 'rb') as gender_pca:    
    pca_gender_90 = pickle.load(gender_pca)

with open(age_model_name, 'rb') as age_model:
    SVC_age_model = pickle.load(age_model)

with open(age_pca_name, 'rb') as age_pca:
    pca_age_90 = pickle.load(age_pca)

def model_predict(face, pca_eivector, model):
    np_face = cv2.resize(face, (100, 100))
    plt.imshow(np_face)
    fd = hog(np_face, orientations=8, pixels_per_cell=(8,8),cells_per_block=(1,1),
                       block_norm= 'L2', multichannel=True)
    np_fd_pca = pca_eivector.transform(fd[np.newaxis,:])
    pred_result = model.predict(np_fd_pca)
    return pred_result

font = cv2.FONT_HERSHEY_SIMPLEX
# video_capture = cv2.VideoCapture('X:\\qq\\qq_files\\1492199983\\FileRecv\\427051444-1-208.mp4')
video_capture = cv2.VideoCapture(1)
fpsLimit = 10
startTime = time.time()


while video_capture.isOpened():
    # Capture frame-by-frame

    ret, frame = video_capture.read()

    # nowTime = time.time()       # possiable FPS control
    # if (int(nowTime - startTime)) > fpsLimit:
    #     # do other cv2 stuff....
    #     startTime = time.time() # reset time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_gray = gray[0:0, 0:0]
    #need to be defined earlier incase there are no face detect at the begining

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(180,180),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        # Draw a rectangle around the faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
        face_gray = gray[y:y+h, x:x+w]
        face_color_GBR = frame[y:y+h, x:x+w]
        
        face_color_RGB = cv2.cvtColor(face_color_GBR, cv2.COLOR_BGR2RGB)
        
        face_predit_result = model_predict(face_color_RGB, pca_gender_90, SVC_gender_model)
        if face_predit_result == 1:
            cv2.putText(frame,'Male',(x+w, y+h), font, 2,(255,0,0),5)
        elif face_predit_result == 0:
            cv2.putText(frame,'Female',(x+w, y+h), font, 2,(255,0,0),5)

        age_predict_result = model_predict(face_color_RGB, pca_age_90, SVC_age_model)
        if age_predict_result == 0:
            cv2.putText(frame,'Children',(x, y), font, 2,(255,0,0),5)
        elif age_predict_result == 1:
            cv2.putText(frame,'Teenagers',(x, y), font, 2,(255,0,0),5)
        elif age_predict_result == 2:
            cv2.putText(frame,'Adults',(x, y), font, 2,(255,0,0),5)
        elif age_predict_result == 3:
            cv2.putText(frame,'Elders',(x, y), font, 2,(255,0,0),5)
    
    # eyes = eyeCascade.detectMultiScale(
    #     face_gray,
    #     scaleFactor= 1.16,
    #     minNeighbors=10,
    #     minSize=(30, 30)
    # )

    # for (ex,ey,ew,eh) in eyes:
    #     cv2.rectangle(face_color_GBR,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    #     cv2.putText(frame,'Eye',(x + ex,y + ey), 1, 1, (0, 255, 0), 1)

    # smile = smileCascade.detectMultiScale(face_gray,scaleFactor= 1.16,minNeighbors=35,minSize=(25, 25),flags=cv2.CASCADE_SCALE_IMAGE)
    # for (sx, sy, sw, sh) in smile:
    #     cv2.rectangle(face_color_GBR, (sh, sy), (sx+sw, sy+sh), (255, 0, 0), 2)
    #     cv2.putText(frame,'Smile',(x + sx,y + sy), 1, 1, (0, 255, 0), 1)



    cv2.putText(frame,'Number of Faces : ' + str(len(faces)),(40, 40), font, 1,(255,0,0),2)      
    # Display the resulting frame
    cv2.imshow('Face Recognition AI', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
