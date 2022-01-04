import tensorflow as tf
from tensorflow.keras import models, layers, losses
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils import *
from naivemodel import SimpleModel

tf.set_random_seed(1)
# reads train data and splits it into x,y,z points
df = pd.read_csv("train.csv", header=None)
x = df.iloc[:,0]
y = df.iloc[:,1]
z = df.iloc[:,2]
a = df.iloc[:,3]
input = packTuple(x,y,z)

# plots points
# pointRender(input,np.expand_dims(np.array(a),1)) # pretty jank code but it works

# simple NN with adam optimizer and MSE loss metric
model = SimpleModel()
model.compile(optimizer='Adam',
                loss=losses.MeanSquaredError(),
                metrics=['mean_squared_error'])
history = model.fit(np.array(input),np.array(a), batch_size = 1000, epochs=100)

model.summary()
# plots out training accuracy
plt.plot(history.history['mean_squared_error'], label='train_mean_squared_error')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.yscale('log')
plt.legend(loc='lower right')
plt.show()

# for outputting predictions of model
df = pd.read_csv("test.csv", header=None)
x = df.iloc[:,0]
y = df.iloc[:,1]
z = df.iloc[:,2]
output = packTuple(x,y,z)

results = model.predict(np.array(output))
np.savetxt("test_results.csv", np.c_[results], delimiter=",")
