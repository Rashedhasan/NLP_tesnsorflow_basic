import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
#https://www.tensorflow.org/tutorials/keras/classification
data=keras.datasets.fashion_mnist
(train_images,traine_labels),(test_images,test_labels)=data.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(train_images[7])#shows the images pixel value in 28x28
plt.imshow(train_images[6])#shows the actual image
#plt.show()
plt.imshow(train_images[6],cmap=plt.cm.binary)#images in binary format
plt.show()
print(train_images.shape)#60000 images with 28x28 size each (60000,28,28)
print(len(traine_labels))#total number of labels
print(traine_labels)#each lebels elements
train_images=train_images/255.0 #pixel value 0 to 1 conversion
test_images=test_images/255.0
print(train_images[6])
#flatten means multidimensional list to single dimension list,Dense means fully connected layer,128 neuron and 10 neuron
model=keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),
                       keras.layers.Dense(128,activation="relu"),
                        keras.layers.Dense(10,activation="softmax")
                       ])
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(train_images,traine_labels,epochs=5)
test_loss,test_acc=model.evaluate(test_images,test_labels)
print("tested Acc",test_acc)
#predict how much each images relate with each labels(10)
prediction=model.predict(test_images)
print(prediction)
print(class_names[np.argmax(prediction[0])])#0th images predicted values and find maximum value index
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i],cmap=plt.cm.binary)
    plt.xlabel("actual :"+class_names[test_labels[i]])
    plt.title("predicted : "+class_names[np.argmax(prediction[i])])
    plt.show()