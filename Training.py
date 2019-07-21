import sys
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import callbacks
from keras import backend as k
from keras import regularizers
import time

start = time.time()

DEV = False
argvs = sys.argv
argc = len(argvs)

if argc > 1 and (argvs[1] == "--development" or argvs[1] == "-d"):
  DEV = True

if DEV:
  epochs = 2
else:
  epochs = 50

train_data_path = 'Fractions DataSet/TrainSet'
validation_data_path = 'Fractions DataSet/TestSet'

"""
Parameters
"""
img_width, img_height = 150, 150
batch_size = 16
#samples_per_epoch = 1000
#validation_steps = 300
train_samples = 50
validation_samples = 28
#nb_filters1 = 32
#nb_filters2 = 64
#conv1_size = 3
#conv2_size = 2
#pool_size = 2
classes_num = 6
#lr = 0.0001

if k.image_data_format() == 'channels_first' :
    input_shape=(3,img_width, img_height)
else :
    input_shape=(img_width, img_height, 3)
    

model = Sequential()
model.add(Conv2D(64, (3,3), border_mode ="same", input_shape=input_shape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(32, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), dim_ordering='th'))

model.add(Flatten())
model.add(Dense(64, input_dim = 64 , kernel_regularizer = regularizers.l2(0.01)))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(classes_num, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer= 'rmsprop',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle = True ,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle = True ,
    class_mode='categorical')

"""
Tensorboard log
"""
log_dir = './tf-log/'
tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
cbks = [tb_cb]

model.fit_generator(
    train_generator,
    steps_per_epoch= train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=cbks,
    validation_steps= validation_samples // batch_size )





target_dir = './models/'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
model.save('./models/model.h5')
model.save_weights('./models/weights.h5')
model.evaluate_generator(validation_generator , validation_samples)


#Calculate execution time                          
end = time.time()
dur = end-start

if dur<60:
    print("Execution Time:",dur,"seconds")
elif dur>60 and dur<3600:
    dur=dur/60
    print("Execution Time:",dur,"minutes")
else:
    dur=dur/(60*60)
    print("Execution Time:",dur,"hours")