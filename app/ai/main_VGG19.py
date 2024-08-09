import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import VGG19
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from sklearn.metrics import classification_report, confusion_matrix

# # GPU Configuration
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         print(e)

# TPU Configuration - Faster?
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    print("Running on TPU:", tpu.master())
except ValueError:
    strategy = tf.distribute.get_strategy()
print("Number of replicas:", strategy.num_replicas_in_sync)

# Parameters
img_height, img_width = 224, 224  # VGG19 input size
batch_size = 32
epochs = 12
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
    fill_mode='nearest',
    validation_split=0.2
)

train_gen = train_data.flow_from_directory(
    './models/testing',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validate_gen = train_data.flow_from_directory(
    './models/validation',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Create tf.data.Dataset objects
def create_dataset(generator):
    return tf.data.Dataset.from_generator(
        lambda: generator,
        output_signature=(
            tf.TensorSpec(shape=(None, img_height, img_width, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, generator.num_classes), dtype=tf.float32)
        )
    ).repeat()

train_ds = create_dataset(train_gen)
validate_ds = create_dataset(validate_gen)

# Disable AutoShard
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
train_ds = train_ds.with_options(options)
validate_ds = validate_ds.with_options(options)

# Save class labels
class_indices = train_gen.class_indices
species_labels = {v: k for k, v in class_indices.items()}
with open('species_labels.json', 'w') as f:
    json.dump(species_labels, f)

# Model building function
def build_model():
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    for layer in base_model.layers[:19]:
        layer.trainable = False
    
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(train_gen.num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model

# Compile model within strategy scope
with strategy.scope():
    model = build_model()
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=0.000001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')

# Training
history = model.fit(
    train_ds,
    steps_per_epoch=train_gen.samples // batch_size,
    epochs=epochs,
    validation_data=validate_ds,
    validation_steps=validate_gen.samples // batch_size,
    callbacks=[reduce_lr, early_stopping, checkpoint]
)

# Save final model
model.save('final_model.keras')

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

# Model Evaluation
predictions = model.predict(validate_ds)
y_pred = np.argmax(predictions, axis=1)
y_true = np.concatenate([y for x, y in validate_ds], axis=0)
y_true = np.argmax(y_true, axis=1)

print(classification_report(y_true, y_pred, target_names=list(species_labels.values())))
print(confusion_matrix(y_true, y_pred))

# Visualization of some predictions
def plot_predictions(x, y_true, y_pred, n=5):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(x[i])
        plt.title(f"True: {species_labels[y_true[i]]}\nPred: {species_labels[y_pred[i]]}")
        plt.axis('off')
    plt.show()

# Get a batch of validation data
x_batch, y_batch = next(iter(validate_gen))
y_pred_batch = model.predict(x_batch)
y_pred_batch = np.argmax(y_pred_batch, axis=1)
y_true_batch = np.argmax(y_batch, axis=1)

# Plot some correct and incorrect predictions
plot_predictions(x_batch, y_true_batch, y_pred_batch)