import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
import json

# Running/Training in parallel to speed up processes
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)

# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         print(e)

# # Parameters for image 
# img_height, img_width =  224, 224   # image dimensions
# batch_size = 32 * len(gpus)    # choose batch size for training
# epochs = 20     # number of training epochs

# # Data generators
# train_data = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     validation_split=0.2
# )

# # Feeding batches of images and corresponding labels to model during training
# # Also path to dataset
# train_gen = train_data.flow_from_directory(
#     './models/flowers', # put path here
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='categorical',
#     subset='training'
# )

# validate_gen = train_data.flow_from_directory(
#     './models/flowers', # put path here
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='categorical',
#     subset='training'
# )

# # Create directory to map class incides to species name
# class_indices = train_gen.class_indices
# species_labels = {v: k for k, v in class_indices.items()}

# # Save species label with JSON
# with open('species_labels.json', 'w') as f:
#     json.dump(species_labels, f)

# # Architecture of model 
# def build_model(num_classes):
#     model = models.Sequential([
#         layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
#         layers.MaxPooling2D((2, 2)),
#         layers.Conv2D(64, (3, 3), activation='relu'),
#         layers.MaxPooling2D((2, 2)),
#         layers.Conv2D(64, (3, 3), activation='relu'),
#         layers.MaxPooling2D((2, 2)),
#         layers.Conv2D(64, (3, 3), activation='relu'),
#         layers.MaxPooling2D((2, 2)),
#         layers.Flatten(),
#         layers.Dense(64, activation='relu'),
#         layers.Dense(num_classes, activation='softmax')
#     ])

#     return model

# # Compile a GPU-model
# strategy = tf.distribute.MirroredStrategy()
# with strategy.scope():
#     model = build_model(len(species_labels))
#     model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# # Training model
# history = model.fit(
#     train_gen,
#     steps_per_epoch=train_gen.samples // batch_size,
#     epochs=epochs,
#     validation_data=validate_gen,
#     validation_steps=validate_gen.samples // batch_size
# )

# # Save model
# model.save('mushroom_plant_classifier.h5')

# # Plot results 
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.legend()
# plt.title('Model Accuracy')
# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.legend()
# plt.title('Model Loss')
# plt.show()

# # Loads image, preprocesses, and uses trained model to predict species -> prints results and confidence score
# def predict_species(image_path):
#     img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = tf.expand_dims(img_array, 0)
    
#     predictions = model.predict(img_array)
#     score = tf.nn.softmax(predictions[0])
    
#     predicted_class = train_gen.class_indices
#     predicted_class = {v: k for k, v in predicted_class.items()}
    
#     print(f"This image most likely belongs to {predicted_class[np.argmax(score)]} with a {100 * np.max(score):.2f} percent confidence.")
    
#     # Also prints out the top 5 predicitons...
#     #...