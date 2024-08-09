import tensorflow as tf
import os
import json
import numpy as np
from tensorflow.keras.models import load_model

model_path = 'v7_flower.keras'
img_dimensions = 240
img_height, img_width =  img_dimensions, img_dimensions  # image dimensions

if os.path.isfile(model_path):
    print("File exists")
    model = load_model(model_path)
else:
    print("File not found")
    
model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


