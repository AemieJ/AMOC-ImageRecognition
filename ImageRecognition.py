import os 
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image 
from keras.preprocessing.image import ImageDataGenerator 

from keras.models import Model 

#Importing our model of known weights that we will use for our transfer learning with CNN
base_model = ResNet50 ( 
        weights = "imagenet" , 
        include_top = False ,  #We dont include the last layer of the ResNet50 model
        input_shape=(128,128,3) 
        )
#Step for making our entire dense layer and output softmax activation function
def build_model(base_model,fc_layers,num_classes,dropout) : 
    x=base_model.output 
    x=Flatten()(x)
    for fc in fc_layers : 
        x=Dense(fc,activation="relu")(x) 
        x=Dropout(dropout)(x) 
    prediction=Dense(num_classes,activation="softmax")(x)
    model=Model(inputs = base_model.input , outputs = prediction )
    return model 

num_classes=len(os.listdir("C:/ml_project/ObjectCategories_sets/TestSet"))
fc_layers=[256,512,512,512,256] 
dropout=0.3 
model=build_model(base_model,fc_layers,num_classes,dropout) 

for layer in model.layers : 
    layer.trainable=False
        
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

train_datagen = ImageDataGenerator(rescale=1./255,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True)
test_datagen  =  ImageDataGenerator(rescale= 1./255)

#Creating the training and test set
training_set = train_datagen.flow_from_directory("C:/ml_project/ObjectCategories_sets/TrainingSet",
                                                 target_size=(128,128),
                                                 color_mode='rgb',
                                                 batch_size= 32 ,
                                                 class_mode="categorical")
test_set = test_datagen.flow_from_directory("C:/ml_project/ObjectCategories_sets/TestSet",
                                            target_size=(128,128),
                                            color_mode='rgb',
                                            batch_size=32 ,
                                            class_mode="categorical")
#Fitting the images to our CNN network created
history=model.fit_generator(training_set,
                   steps_per_epoch=250,
                   epochs=25,
                   validation_data=test_set ,
                   validation_steps=63)

import matplotlib.pyplot as plt 
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

import numpy as np
test_image = image.load_img("",
                            target_size=(64,64))
test_image = image.img_to_array(test_image) #64x64
test_image= np.expand_dims(test_image,axis=0) #64x64x1
result = model.predict_classes(test_image)
training_set.class_indices