#!/usr/bin/env python
# coding: utf-8

# # Pre-Processing
# ![preprocesing-wajah](https://user-images.githubusercontent.com/60698877/107317690-b7d59580-6acd-11eb-9a83-2211e36c41ea.JPG)

# In[1]:


import matplotlib.pyplot as plt
import cv2,os

data_path='C:/Users/MirnaF/Latihan-1/DB-FIX/DB2-lengkap/data-targetDB2'
categories=os.listdir(data_path)
labels=[i for i in range(len(categories))]

label_dict=dict(zip(categories,labels))

print(label_dict)
print(categories)
print(labels)


# In[2]:


img_size=100

data=[]
target=[]


for category in categories:
    folder_path=os.path.join(data_path,category)
    img_names=os.listdir(folder_path)   #alamat si FOLDER
    print (folder_path)
    print ('--------------------------------------')
    print (img_names)
    print ('**************************************')
    
    for img_name in img_names:
        img_path=os.path.join(folder_path,img_name)
        img=cv2.imread(img_path)

        try:
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)           
            #Coverting the image into gray scale
            
            resized=cv2.resize(gray,(img_size,img_size)) #untuk wajah, kan square
            #resizing the gray scale into 100x100, since we need a fixed common size for all the images in the dataset
            
            data.append(resized)
            target.append(label_dict[category])
            #appending the image and the label(categorized) into the list (dataset)
  
        except Exception as e:
            print('Exception:',e)
            #if any exception rasied, the exception will be printed here. And pass to the next image


# In[3]:


import numpy as np
data=np.array(data)/255.0
print(data.shape)


# In[4]:


#mereshape image yg ada dalam data (berupa array)

data=np.reshape(data,(data.shape[0],img_size,img_size,1)) #untuk yg wajah (square)

target=np.array(target)

from keras.utils import np_utils

new_target=np_utils.to_categorical(target)


# In[5]:


np.save('dataDB2',data)
np.save('targetDB2',new_target)


# In[7]:


print(data.shape)


# In[ ]:




