import tensorflow as tf
import os
import json
import numpy as np

model_path = 'v4_flowers.keras'
img_height, img_width =  224, 224   # image dimensions
model = (model_path)

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


def predict_species(image_path, i):
    # Load the species labels
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
     
    try:
        with open('species_labels.json', 'r') as f:
            species_labels = json.load(f)
    except FileNotFoundError:
        print("Error: species_labels.json not found. Make sure to train the model first.")
        return
    
    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0  # Normalize the image

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = np.argmax(score)
    predicted_species = species_labels[str(predicted_class)]
    confidence = 100 * np.max(score)

    print(f"This image ({i}) most likely belongs to the species: {predicted_species}")
    print(f"Confidence: {confidence:.2f}%")

    top_5_indices = np.argsort(score)[-5:][::-1]
    print("\nTop 5 predictions:")
    for i in top_5_indices:
        species = species_labels[str(i)]
        confidence = 100 * score[i]
        print(f"{species}: {confidence:.2f}%")


NUMBER_OF_TESTS = 10

for i in range(NUMBER_OF_TESTS):
    predict_species(f'test/{i + 1}.jpg', i + 1)
