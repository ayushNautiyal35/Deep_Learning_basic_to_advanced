

import pandas as pd

df=pd.read_csv("C:/Users/ayush/Downloads/archive/Churn_Modelling.csv")

df.head()

x=df.iloc[:,:-1]

x

y=df.iloc[:,-1]

y

x.drop('RowNumber',axis=1,inplace=True)
x.drop('CustomerId',axis=1,inplace=True)
x.drop('Surname',axis=1,inplace=True)

x



x['Geography']=x['Geography'].map({'France':0,'Spain':1,'Germany':2})

x['Gender']=x['Gender'].map({'Male':0,'Female':1})

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(x_train)
sc.fit(x_test)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout

classifier=Sequential()

#Adding input and hidden layer
classifier.add(Dense(units=6,kernel_initializer='he_normal',activation='relu',input_dim=10))

#adding second hidden layer
classifier.add(Dense(units=6,kernel_initializer='he_normal',activation='relu'))

#adding output layer
classifier.add(Dense(units=1,kernel_initializer='glorot_uniform',activation='sigmoid'))

#compiling the ann
classifier.compile(optimizer='Adamax',loss='binary_crossentropy',metrics=['accuracy'])

#fitting the ann to the training set
model_history=classifier.fit(x_train,y_train,validation_split=0.33,batch_size=10,epochs=100)


#print all data
print(model_history.history.keys())

#summarize history for loss
import matplotlib.pyplot as plt

plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'])
plt.show()

#prediction the test set results
y_pred=classifier.predict(x_test)
y_pred=(y_pred>0.5)

#make confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#calcuate the accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred, y_test)

print(score)







