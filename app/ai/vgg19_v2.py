import matplotlib.pyplot as plt
# import seaborn as sns
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications import VGG19, ResNet50
from keras.callbacks import ReduceLROnPlateau
import cv2
import os
import random
import tensorflow as tf
import numpy as np
import pandas

labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
img_size = 224
batchSize = 32
numOfEpochs = 12

def get_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try: 
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data, dtype=object)

train_path = './models/testing'
# validate_path = './models/validation'

trainingData = get_data(train_path)
# validationData = get_data(validate_path)

x = []
y = []

for feature, label in trainingData:
    x.append(feature)
    y.append(label)

# validation_x = []
# validation_y = []

# for feature, label in validationData:
#     validation_x.append(feature)
#     validation_y.append(label)

x = np.array(x) / 255
x = x.reshape(-1, img_size, img_size, 3)

y = np.array(y)

label_binarizer = LabelBinarizer()
y = label_binarizer.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split( x , y , test_size = 0.2 , stratify = y, random_state = 0 )

del x,y,trainingData

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    pre_trained_model = VGG19(input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet")

    for layer in pre_trained_model.layers[:19]:
        layer.trainable = False

    model = Sequential([
        pre_trained_model,
        MaxPool2D((2,2), strides = 2),
        Flatten(),
        Dense(5, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose = 1, factor = 0.3, min_lr = 0.000001)
history = model.fit(x_train, y_train, batch_size = batchSize, epochs = numOfEpochs, validation_data = (x_test, y_test), callbacks = [learning_rate_reduction])

print("Loss of the model is - " , model.evaluate(x_test,y_test)[0])
print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")


epochs = [i for i in range(12)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
fig.set_size_inches(20,10)

ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
ax[0].plot(epochs , val_acc , 'ro-' , label = 'Testing Accuracy')
ax[0].set_title('Training & Testing Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
ax[1].plot(epochs , val_loss , 'r-o' , label = 'Testing Loss')
ax[1].set_title('Training & Testing Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
plt.show()

model.save('first_vgg19_v2.keras')