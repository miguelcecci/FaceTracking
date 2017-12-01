import numpy as np
import cv2
import io
import random
from time import sleep
from skimage import feature, exposure

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def lbp_histogram(imagem, eps, numero_pontos, raio):
    lbp = feature.local_binary_pattern(imagem, numero_pontos, raio, method="uniform")
    (histograma, _) = np.histogram(lbp.ravel(), bins=np.arange(0, numero_pontos+10), range=(0, numero_pontos+2))
    histograma = histograma.astype("float")
    histograma /= (histograma.sum() + eps)
    return histograma

cap = cv2.VideoCapture(0)
cAvg = 5.48e+01 #media de valores de variaçao de cor extraidos previamente
lAvg = 1.18e-04 #idem lbp
dAvg = 0.12 # idem distancia

lbp_faces = []

def getRandCollor():
    return((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),2)
        crop = frame[y:y+h, x:x+w]
        avg_color = [np.floor(crop[:, :, w].mean()) for w in range(crop.shape[-1])]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lbp = lbp_histogram(crop, 1e-7, 24, 8)/(h*w)
        if len(lbp_faces) == 0:
            lbp_faces.append([lbp, x, y, avg_color, getRandCollor()])
            print('Rosto Inserido!')
        else:
            aux = [10000, -1]
            for i, lbp_face in enumerate(lbp_faces):
                lbpdist = np.linalg.norm(lbp_face[0]-lbp)
                colordist = np.linalg.norm(np.array(lbp_face[3])-np.array(avg_color))
                eucldist = np.linalg.norm(np.array(lbp_face[1], lbp_face[2]) - np.array(x, y))
                # print(lbpdist, "   ", round(colordist, 4), "   ", eucldist)
                coefvar = (lbpdist/lAvg)*(colordist/cAvg)*(eucldist/dAvg) #coeficiente de variacao
                if(coefvar < aux[0]):
                    aux = [coefvar, i]
            if(aux[0] < 10): #esse limite 10 foi baseado em chute
                cor_do_retanglinho = lbp_faces[aux[1]][4]
                lbp_faces[aux[1]] = [lbp, x, y, avg_color, cor_do_retanglinho]
                cv2.rectangle(frame,(x,y),(x+w,y+h),lbp_faces[aux[1]][4],4)
            else:
                lbp_faces.append([lbp, x, y, avg_color, getRandCollor()])
                print('Rosto Inserido!')
    cv2.imshow('img',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
