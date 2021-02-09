#!/usr/bin/env python
# coding: utf-8

# # Convolutional Neural Network Architecture
# ![arsitektur CNN wajah-FACE](https://user-images.githubusercontent.com/60698877/107318115-97f2a180-6ace-11eb-9089-bdb4d5a1ad23.png)

# In[1]:


import time
start = time.time()


# In[2]:


import numpy as np

data=np.load('targetDB2.npy')
target=np.load('targetDB2.npy')


# In[3]:


data.shape[1:]


# In[4]:


from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint

model=Sequential()

model.add(Conv2D(300,(3,3),input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
#The first CNN layer followed by Relu and MaxPooling layers

model.add(Conv2D(200,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
#The second convolution layer followed by Relu and MaxPooling layers

model.add(Conv2D(150,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
#The second convolution layer followed by Relu and MaxPooling layers

model.add(Flatten())
model.add(Dropout(0.5))
#Flatten layer to stack the output convolutions from second convolution layer
model.add(Dense(128,activation='relu'))
#Dense layer of 64 neurons
model.add(Dense(6,activation='softmax'))
#The Final layer with two outputs for 10 categories
# kalo yg data dan target dia 9 kategori --> dense 9
# kalo yg dataDB2 dan targetDB2 dia 10 kategori --> dense 10
# kalo yg dataROI dan targetROI dia 5 kategori --> dense 5

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[5]:


from sklearn.model_selection import train_test_split

train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.1)
print(train_data.shape, test_data.shape)


# In[6]:


checkpoint = ModelCheckpoint('modelDB2-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
history=model.fit(train_data,train_target,epochs=40,callbacks=[checkpoint],validation_split=0.2)


# In[7]:


# untuk melihat ringkasan model yg dibuat. 
# tau brp jmlh parameter yg digunakan

model.summary()


# In[8]:


from matplotlib import pyplot as plt

plt.plot(history.history['loss'],'r',label='training loss')
plt.plot(history.history['val_loss'],label='validation loss')
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# In[9]:


plt.plot(history.history['accuracy'],'r',label='training accuracy')
plt.plot(history.history['val_accuracy'],label='validation accuracy')
plt.xlabel('# epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[22]:


print(test_data.shape, test_target.shape)


# In[10]:


print(model.evaluate(test_data,test_target))


# In[11]:


end = time.time()
print("Execution Time (in seconds) :")
print("Trainning Time : ", format(end - start, '.2f'))


# # Confusion Matrix

# In[13]:


y_pred = model.predict(test_data)
y_pred = np.argmax(y_pred, axis=1)
test_target = np.argmax(test_target, axis=1)


# In[21]:


total = 0
accurate = 0
accurateindex = []
wrongindex = []
result=model.predict(test_data)

for i in range(len(result)):
    if np.argmax(y_pred[i]) == np.argmax(test_target[i]):
        accurate += 1
        accurateindex.append(i)
    else:
        wrongindex.append(i)
        
    total += 1
    
print('Total-test-data;', total, '\taccurately-predicted-data:', accurate, '\t wrongly-predicted-data: ', total - accurate)
print('Accuracy:', round(accurate/total*100, 3), '%')


# In[16]:


#importing confusion matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(test_target, y_pred)
print('Confusion Matrix\n')
print(confusion)


# In[17]:


#ini berhasil buat nampilin lebih menarik

import seaborn as sns
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=(8,5))
sns.heatmap(confusion, annot=True, fmt=".0f", ax=ax, cmap=plt.cm.Blues)
plt.tight_layout()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[18]:


FP = confusion.sum(axis=0) - np.diag(confusion) 
FN = confusion.sum(axis=1) - np.diag(confusion)
TP = np.diag(confusion)
TN = confusion.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
# Overall accuracy for each class
ACC = (TP+TN)/(TP+FP+FN+TN)


# In[19]:


print ("TPR = ", TPR)
print ("TNR = ", TNR)
print ("PPV = ", PPV)
print ("NPV = ", NPV)
print ("FPR = ", FPR)
print ("FNR = ", FNR)
print ("FDR = ", FDR)
print ("ACC = ", ACC)

#ini untuk tau nilai setiap output 0 1 2 3 4 dst (kepinggir)


# In[20]:


from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(test_target, y_pred))


# In[ ]:





# In[ ]:




