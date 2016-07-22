
# coding: utf-8

# In[1]:

#import dataset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

training_data = np.genfromtxt('./training_data/complexIntersectionTrainingData.txt')
x_train, y_train = training_data[:,:4], training_data[:,4:]


# In[2]:

y_train[0:10]


# In[3]:

from keras.models import Sequential
from keras.layers.core import Dense

model = Sequential()
model.add(Dense(input_dim=x_train.shape[1], 
                output_dim=100, 
                init='uniform', 
                activation='tanh'
               )
         )

model.add(Dense(output_dim=y_train.shape[1], 
                init='uniform', 
                activation='softmax',
               )
         )

model.compile(loss='mse', optimizer='Adagrad')


# In[4]:

#train on dataset
h = model.fit(x_train/1200.0, y_train, batch_size=32, validation_split=0.1, nb_epoch=100, verbose=0)


# In[5]:

# plot error loss
plt.plot(h.history['loss'], c='b')
plt.plot(h.history['val_loss'], c='g')
plt.show()

