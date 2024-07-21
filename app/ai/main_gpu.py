import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import os
import json

# Running/Training in parallel to speed up processes
gpus = tf.config.experimental.list_physical_devices('GPU')

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# print("GPU Details:", tf.config.list_physical_devices('GPU'))

if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

# print(len(gpus))

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Parameters for image 
img_height, img_width =  300, 300  # image dimensions
batch_size = 32     # choose batch size for training
epochs = 50     # number of training epochs
tf.random.set_seed(42)

# Data generators
train_data = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Feeding batches of images and corresponding labels to model during training
# Also path to dataset
train_gen = train_data.flow_from_directory(
    './models/testing', # put path here
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validate_gen = train_data.flow_from_directory(
    './models/validation', # put path here
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

train_ds = tf.data.Dataset.from_generator(
    lambda: train_gen,
    output_signature=(
        tf.TensorSpec(shape=(None, img_height, img_width, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, train_gen.num_classes), dtype=tf.float32)
    )
).repeat()

validate_ds = tf.data.Dataset.from_generator(
    lambda: validate_gen,
    output_signature=(
        tf.TensorSpec(shape=(None, img_height, img_width, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, validate_gen.num_classes), dtype=tf.float32)
    )
).repeat()

# Create options object
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

# Apply options to both datasets
train_ds = train_ds.with_options(options)
validate_ds = validate_ds.with_options(options)

# Create directory to map class incides to species name
class_indices = train_gen.class_indices
species_labels = {v: k for k, v in class_indices.items()}

# Save species label with JSON
with open('species_labels.json', 'w') as f:
    json.dump(species_labels, f)

# # Architecture of model 
# def build_model():
#     model = models.Sequential([
#         layers.Input(shape=(img_height, img_width, 3)),
#         layers.Conv2D(32, (3, 3), activation='relu'),
#         layers.MaxPooling2D((2, 2)),
#         layers.Conv2D(64, (3, 3), activation='relu'),
#         layers.MaxPooling2D((2, 2)),
#         layers.Conv2D(64, (3, 3), activation='relu'),
#         layers.MaxPooling2D((2, 2)),
#         layers.Conv2D(64, (3, 3), activation='relu'),
#         # layers.MaxPooling2D((2, 2)),
#         # layers.Conv2D(128, (3, 3), activation='relu'),  # Added convolutional layer
#         # layers.MaxPooling2D((2, 2)),                    # Added pooling layer
#         # layers.Conv2D(128, (3, 3), activation='relu'),  # Another added convolutional layer
#         layers.MaxPooling2D((2, 2)),        
#         layers.Flatten(),
#         layers.Dense(64, activation='relu'),
#         layers.Dropout(0.3),
#         layers.Dense(train_gen.num_classes, activation='softmax')
#     ])

#     return model

def inception_module(x, filters):
    # 1x1 conv
    conv1x1 = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(x)
    
    # 3x3 conv
    conv3x3 = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    
    # 5x5 conv
    conv5x5 = layers.Conv2D(filters, (5, 5), padding='same', activation='relu')(x)
    
    # 3x3 max pooling
    pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    pool = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(pool)
    
    # Concatenate all branches
    output = layers.Concatenate()([conv1x1, conv3x3, conv5x5, pool])
    return output

def build_model():
    inputs = layers.Input(shape=(img_height, img_width, 3))
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = inception_module(x, 64)
    x = layers.MaxPooling2D((2, 2))(x)
    x = inception_module(x, 128)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(train_gen.num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# Implementing Learning Rate Schedule
def lr_schedule(epoch):
    lr = 0.001
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 150:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    return lr

# Compile a GPU-model
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = build_model()
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    lr_scheduler = LearningRateScheduler(lr_schedule)

    # Early Stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
# Training model
history = model.fit(
    train_ds,
    steps_per_epoch=train_gen.samples // batch_size,
    epochs=epochs,
    validation_data=validate_ds,
    validation_steps=validate_gen.samples // batch_size,
    callbacks=[lr_scheduler, early_stopping]
)

# Save model
model.save('v1_gpu_flowers.keras')

# Plot results 
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')
plt.show()
