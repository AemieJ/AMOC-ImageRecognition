# Object Recognition using Convolutional Neural Networks

Multi class image classifier made using keras and tensorflow implemented on the [Caltech 101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/) Dataset. 

### Getting Started
1. Pull this repository to your local machine
2. Import all the required libraries
3. Add images you want to predic to the ```images``` directory
4. Run the ```predict.py``` file and pass in the image name as an argument
```
python predict.py image.jpg
```

### Importing

1. Python 3
2. Tesorflow
3. Keras
4. Numpy
5. Matplotlib

```
pip install -r requirements.txt
```


### Training Models
Download the [Caltech 101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/) Dataset.

The ```test_train_split.py``` script will split the dataset into training and test dataset in the form of two seperate directories.

Use the ```train.py``` or ```transfer_learning.py``` to train the models. The  ```train.py```
will train the neural network from scratch while ```transfer_learning.py``` will use the
[Inception-v3](https://towardsdatascience.com/how-to-train-your-model-dramatically-faster-9ad063f0f718) model.

## Network Structure
This is the strcuture implemented in the ```predict.py``` file.
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 128, 128, 32)      896       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 64, 64, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 64, 64, 32)        9248      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 32, 32, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 32, 32, 64)        18496     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 16, 16, 64)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 16, 16, 64)        36928     
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 8, 8, 64)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 4096)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 2048)              8390656   
_________________________________________________________________
dropout_1 (Dropout)          (None, 2048)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 2048)              4196352   
_________________________________________________________________
dense_3 (Dense)              (None, 2048)              4196352   
_________________________________________________________________
dropout_2 (Dropout)          (None, 2048)              0         
_________________________________________________________________
dense_4 (Dense)              (None, 2048)              4196352   
_________________________________________________________________
dense_5 (Dense)              (None, 2048)              4196352   
_________________________________________________________________
dense_6 (Dense)              (None, 2048)              4196352   
_________________________________________________________________
dropout_3 (Dropout)          (None, 2048)              0         
_________________________________________________________________
dense_7 (Dense)              (None, 2048)              4196352   
_________________________________________________________________
dense_8 (Dense)              (None, 56)                114744    
=================================================================
Total params: 33,749,080
Trainable params: 33,749,080
Non-trainable params: 0
_________________________________________________________________
```

## Built With

* [Tensoflow](https://tensorflow.org) - Computation Backend
* [Keras](https://keras.io) - Neural Network API
 

## Authors

* **Sahil Bondre**- [GodCrampy](https://github.com/godcrampy)
* **Aemie Jariwala**-- [AemieJ](https://github.com/AemieJ)


## Acknowledgments

* Team ACM for hosting AMOC
* ACM Mentor Mansi
