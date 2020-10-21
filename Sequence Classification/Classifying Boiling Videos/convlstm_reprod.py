import keras
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.utils import multi_gpu_model
#from keras import tensorflow
 
import os
import cv2
import numpy as np
import matplotlib.pyplot
import seaborn as sns
import time
import random as python_random

from sklearn.model_selection import train_test_split
import keras_metrics as km 
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

np.random.seed(123)
python_random.seed(123)
#tensorflow.random.set_seed(1234)

data_dir = "videos/"
img_height , img_width = 128, 128
seq_len = 10
 
classes = ["class_0", "class_1", "class_2"]

num_gpus = 2
 
#  Creating frames from videos
 
def frames_extraction(video_path):
    frames_list = []
     
    vidObj = cv2.VideoCapture(video_path)
    # Used as counter variable 
    count = 1
 
    while count <= seq_len: 
         
        success, image = vidObj.read() 
        if success:
            image = cv2.resize(image, (img_height, img_width))
            frames_list.append(image)
            #print(vidObj.get(7))
            count += 1
        else:
            print("Defected frame")
            break
 
            
    return frames_list
 
def create_data(input_dir):
    X = []
    Y = []
     
    classes_list = sorted(os.listdir(input_dir))
    print(classes_list) 
    for c in classes_list:
        if c == "class_0":
            print(c)
            files_list = os.listdir(os.path.join(input_dir, c))
            for f in files_list:
                frames = frames_extraction(os.path.join(os.path.join(input_dir, c), f))
                if len(frames) == (seq_len):
                    X.append(frames)
                    y = [0]*len(classes)
                    y[classes.index(c)] = 1
                    Y.append(y)
        else:
            pass

    for c in classes_list:
        if c == "class_1":
            print(c)
            files_list = os.listdir(os.path.join(input_dir, c))
            for f in files_list:
                frames = frames_extraction(os.path.join(os.path.join(input_dir, c), f))
                if len(frames) == (seq_len):
                    X.append(frames)
                    y = [0]*len(classes)
                    y[classes.index(c)] = 1
                    Y.append(y)
        else:
            pass

    for c in classes_list:
        if c == "class_2":
            print(c)
            files_list = os.listdir(os.path.join(input_dir, c))
            for f in files_list:
                frames = frames_extraction(os.path.join(os.path.join(input_dir, c), f))
                if len(frames) == (seq_len):
                    X.append(frames)
                    y = [0]*len(classes)
                    y[classes.index(c)] = 1
                    Y.append(y)
        else:
            pass
     
    X = np.asarray(X)
    Y = np.asarray(Y)
    return X, Y

X, Y = create_data(data_dir)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=0)

model = Sequential()
model.add(ConvLSTM2D(filters = 64, kernel_size = (3, 3), return_sequences = False, 
                     data_format = "channels_last", input_shape = (seq_len, img_height, img_width, 3)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(3, activation = "softmax"))
 
model.summary()

if num_gpus > 1:
	parallel_model = multi_gpu_model(model, gpus=num_gpus)
elif num_gpus == 1:
	parallel_model = model

opt = keras.optimizers.SGD(lr=0.001)
parallel_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])

# Prepare callback for training time.
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times=[]
    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start=time.time()
    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

time_callback = TimeHistory()

earlystop = EarlyStopping(patience=7)
mc = ModelCheckpoint('mdl_weights_1.1.7.hdf5', save_best_only=True, monitor='val_loss', mode='min')
callbacks = [earlystop, time_callback] # removed modelcheckpoint for scalability

history = parallel_model.fit(x = X_train, y = y_train, epochs=5, batch_size = 8*num_gpus, 
                    shuffle=True, validation_split=0.2, callbacks=callbacks,
                    verbose=1)

loss = history.history['loss']
val_loss = history.history['val_loss']
accuracy = history.history['acc']
val_accuracy = history.history['val_acc']

outputfilename = r"output_reprod_" + str(num_gpus) + "gpu.txt"
output_file = open(outputfilename, "w")

output_file.write("loss\n")
for i in range(len(loss)):
    output_file.write(str(loss[i]))
    output_file.write("\n")
output_file.write("\nval_loss\n")
for i in range(len(val_loss)):
    output_file.write(str(val_loss[i]))
    output_file.write("\n")
output_file.write("\naccuracy\n")
for i in range(len(accuracy)):
    output_file.write(str(accuracy[i]))
    output_file.write("\n")
output_file.write("\nval_accuracy\n")
for i in range(len(val_accuracy)):
    output_file.write(str(val_accuracy[i]))
    output_file.write("\n")
times = time_callback.times
output_file.write("\nTotal time\n")
output_file.write(str(np.sum(times)))
output_file.write("\nAverage time\n")
output_file.write(str(np.sum(times)/len(times)))
output_file.close()

times = time_callback.times
print("Total time: ", np.sum(times))
print("Average time: ", np.sum(times)/len(times))
