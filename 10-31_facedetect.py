import pickle
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from skimage.feature import hog

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

model_name = 'svc_model_v1.1.h5'
pca_name = 'pca_90_eivector.pca'

with open(model_name, 'rb') as model:
    SVC_model = pickle.load(model)

with open(pca_name, 'rb') as pca:    
    pca_90 = pickle.load(pca)


def gender_predict(face, pca_eivector, model):
    np_face = cv2.resize(face, (100, 100))
    plt.imshow(np_face)
    fd = hog(np_face, orientations=8, pixels_per_cell=(8,8),cells_per_block=(1,1),
                       block_norm= 'L2', multichannel=True)
    np_fd_pca = pca_eivector.transform(fd[np.newaxis,:])
    pred_result = model.predict(np_fd_pca)
    return pred_result

font = cv2.FONT_HERSHEY_SIMPLEX
video_capture = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(150,150),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        roi_color = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
        
        face_predit_result = gender_predict(roi_color, pca_90, SVC_model)
        if face_predit_result == 1:
            cv2.putText(frame,'Male Face',(x, y), font, 2,(255,0,0),5)
        elif face_predit_result == 0:
            cv2.putText(frame,'Female Face',(x, y), font, 2,(255,0,0),5)

    # smile = smileCascade.detectMultiScale(roi_gray,scaleFactor= 1.16,minNeighbors=35,minSize=(25, 25),flags=cv2.CASCADE_SCALE_IMAGE)
    # for (sx, sy, sw, sh) in smile:
    #     cv2.rectangle(roi_color, (sh, sy), (sx+sw, sy+sh), (255, 0, 0), 2)
    #     cv2.putText(frame,'Smile',(x + sx,y + sy), 1, 1, (0, 255, 0), 1)

    # eyes = eyeCascade.detectMultiScale(roi_gray,scaleFactor= 1.16,minNeighbors=20,minSize=(30, 30))
    # for (ex,ey,ew,eh) in eyes:
    #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    #     cv2.putText(frame,'Eye',(x + ex,y + ey), 1, 1, (0, 255, 0), 1)

    cv2.putText(frame,'Number of Faces : ' + str(len(faces)),(40, 40), font, 1,(255,0,0),2)      
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
