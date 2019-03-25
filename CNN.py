# creating dictionary for set of objects
import os
obj = dict()
directory = "C:\ObjectCategories"
filename = os.listdir(directory)
for iter in range(len(filename)) :
    obj[iter] =  filename[iter]

from keras.models import  Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout


#CREATING A NEURAL NETWORK TO WORK ON
classifier = Sequential()
#Creating  input layer

#Step 1 : to apply convolution and relu
classifier.add(Conv2D(filters=32 ,kernel_size = (3,3),activation = "relu" ,data_format="channels_last",input_shape=(64,64,3)))

#Step 2 : to apply maxpooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step 3 : apply convolution , relu and max pooling 2-4 times
classifier.add(Conv2D(filters=32 ,kernel_size = (3,3),activation = "relu" ,data_format="channels_last",input_shape=(64,64,3)))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Conv2D(filters=32 ,kernel_size = (3,3),activation = "relu" ,data_format="channels_last",input_shape=(64,64,3)))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step 4 : to flatten the matrix into a column vector
classifier.add(Flatten())

#Creating hidden layer
classifier.add(Dense(units=120,activation="relu"))
classifier.add(Dropout(rate=0.25))
#classifier.add(Dense(units=80,activation="relu"))
classifier.add(Dense(units=120,activation="relu"))
classifier.add(Dropout(rate=0.25))

#Creating the output fully connected layer
classifier.add(Dense(units=101,activation="softmax"))

#Compiling and optimizing the entire cnn ; we are going to use stochaistic GD
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])


#TRAINING OUR TRAIN AND CROSS VAL AND TEST SET
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True)
test_datagen  =  ImageDataGenerator(rescale= 1./255)

#Creating the training and test set
training_set = train_datagen.flow_from_directory("C:/ml_project/ObjectCategories_sets/training_set",
                                                 target_size=(64,64),
                                                 batch_size= 32 ,
                                                 class_mode="binary")
test_set = test_datagen.flow_from_directory("C:/ml_project/ObjectCategories_sets/test_set",
                                            target_size=(64,64),
                                            batch_size=32 ,
                                            class_mode="binary")
#Fitting the images to our CNN network created

classifier.fit_generator(training_set ,
                         steps_per_epoch=250 ,
                         epochs=25 ,
                         validation_data = test_set ,
                         validation_steps=63)

#Training our model

import numpy as np
from keras_preprocessing import image
test_image = image.load_img("",
                            target_size=(64,64))
test_image = image.img_to_array(test_image) #64x64
test_image= np.expand_dims(test_image,axis=0) #64x64x1
result = classifier.predict_classes(test_image)
training_set.class_indices

for i in range(len(result)) :
    if result[i][0] == 1 :
        prediction=obj[i]













