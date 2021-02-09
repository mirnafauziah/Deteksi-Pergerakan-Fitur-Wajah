#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import sys
import cv2
import dlib
from math import hypot
import numpy as np
from keras.models import load_model


# In[3]:


kameraVideo = cv2.VideoCapture(0)
if not kameraVideo.isOpened():
    print ('Kamera Tidak Dapat Diakses')
    exit()

#menampilkan informasi video (script 2 di BAB 13)-----------------------------------------------------------
else:
    tinggi = kameraVideo.get(cv2.CAP_PROP_FRAME_HEIGHT)
    lebar = kameraVideo.get(cv2.CAP_PROP_FRAME_WIDTH)
    jumFrame = kameraVideo.get(cv2.CAP_PROP_FPS)
    
    print('Tinggi:', tinggi)
    print('Lebar:', lebar)
    print('Jumlah Frame per Second:', jumFrame)


# In[4]:


import logging
import sys
import os

def setup_custom_logger():
    LOG_DIR = os.getcwd() + '/' + 'logs'
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler(LOG_DIR+'/log.txt', mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger


# In[5]:


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

model_F = load_model('modelDB2-037.model')

#khusus modelDB2-037 (ada 9 kelas)
labels_dict_F={0:'Mengerutkan Dahi',
             1:'Mengangkat Kedua Alis',
             2:'Lirik Kanan',
             3:'Lirik Kiri',
             4:'Normal',
             5:'Lihat Atas',
             6:'Lihat Bawah',
             7:'Lihat Kanan',
             8:'Lihat Kiri',
             9:'Lirik Bawah'}
color_dict_F={0:(236,186,136),
              1:(255,204,255),
              2:(0,0,128),
              3:(211,85,186),
              4:(0,255,0),
              5:(0,155,255),
              6:(255,255,0),
              7:(255,0,0),
              8:(128,128,0), 
              9:(135,184,222)}


# In[6]:


index = 0
logger = setup_custom_logger()

while (True):
    try:
        #ambil per KERANGKA UNTUK DIPROSES
        ret, kerangkaAsal = kameraVideo.read()

        #siap2 ngesave dgn judul berdasarkan counter index
        if not ret: 
            break

        dets = detector(kerangkaAsal)

        num_faces = len(dets)
        #print("banyaknya wajah:",num_faces)

        if num_faces == 0:
            print("Sorry, there were no faces found")
            logger.debug('Face Not Found')
            cv2.putText(kerangkaAsal, "NO FACES", (200,240), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 5)

        # Find the 5 face landmarks we need to do the alignment.
        faces = dlib.full_object_detections()

        for detection in dets:
            # predictor tuh sp asalnya
            faces.append(predictor(kerangkaAsal, detection))

        images = dlib.get_face_chips(kerangkaAsal, faces, size=320)

        for image in images:

            #konversi ke skala abu-abu
            kerangkaAbu = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            #cv2.imshow('Kerangka Abu-abu', kerangkaAbu)

    #bila index habis dibagi 5 (incase mau diambil tiap frame ke 5)
        if index%5==0:
            #isi dengan kode apa yg mau dilakukan saat mendeteksi frame ke 5
            
            resized_F=cv2.resize(kerangkaAbu,(100,100))
            normalized_F=resized_F/255.0
            reshaped_F=np.reshape(normalized_F,(1,100,100,1))
            result_F=model_F.predict(reshaped_F)
                                  
        index += 1
        label_F=np.argmax(result_F,axis=1)[0]
        #cv2.imshow('REGION FACE',image)
        #print(labels_dict_F[label_F])
        logger.info(labels_dict_F[label_F])
        
        if (label_F == 6) or (label_F == 9):
            cv2.rectangle(kerangkaAsal,(10,400),(200,450),color_dict_F[6],-1)
            cv2.putText(kerangkaAsal, labels_dict_F[6], (10,430),cv2.FONT_HERSHEY_SIMPLEX,0.8, (255,255,255),2)
                   
        #untuk tampilan saja
        elif (label_F != 0) & (label_F != 1):        
            cv2.rectangle(kerangkaAsal,(10,400),(200,450),color_dict_F[label_F],-1)
            cv2.putText(kerangkaAsal, labels_dict_F[label_F], (20,430),cv2.FONT_HERSHEY_SIMPLEX,0.8, (255,255,255),2)

        else: 
        #ukuran latar untuk mengangkat kedua alis
            cv2.rectangle(kerangkaAsal,(10,400),(350,450),color_dict_F[label_F],-1)
            cv2.putText(kerangkaAsal, labels_dict_F[label_F], (10,430),cv2.FONT_HERSHEY_SIMPLEX,0.8, (255,255,255),2)

    except RuntimeError as e:
        print (e)
    
    #Tampilkan
    cv2.imshow('Cheating Detection', kerangkaAsal)
    #cv2.imshow('Kerangka Asal', image)
        
    key=cv2.waitKey(1)

    if(key==27):
        print ("-------------------Program Selesai Digunakan--------------------")
        break
        
cv2.destroyAllWindows()
kameraVideo.release()        


# In[ ]:




