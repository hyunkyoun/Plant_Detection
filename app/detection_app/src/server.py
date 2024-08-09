import tensorflow as tf
import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, jsonify, request
from flask_cors import CORS

model_path = 'v7_flower.keras'

if os.path.isfile(model_path):
    print("File exists")
    model = load_model(model_path)
else:
    print("File not found")
    
model.summary()
model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])    

app = Flask(__name__)
CORS(app)

def predict_species(image_path):
    img_dimensions = 240
    img_height, img_width =  img_dimensions, img_dimensions  # image dimensions'

    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return "unknown", 0.0
     
    try:
        with open('species_labels.json', 'r') as f:
            species_labels = json.load(f)
    except FileNotFoundError:
        print("Error: species_labels.json not found. Make sure to train the model first.")
        return "unknown", 0.0
    
    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) / 255.0  # Normalize the image
    except Exception as e:
        print(f"Error loading image: {e}")
        return "unknown", 0.0
    
    try:
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        predicted_class = np.argmax(score)
        predicted_species = species_labels[str(predicted_class)]
        confidence = 100 * np.max(score)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "unknown", 0.0

    print(f"\nThis image most likely belongs to the species: {predicted_species}")
    print(f"Confidence: {confidence:.2f}%")
    return predicted_species, confidence

    # top_5_indices = np.argsort(score)[-5:][::-1]
    # print("\nTop 5 predictions:")
    # for i in top_5_indices:
    #     species = species_labels[str(i)]
    #     confidence = 100 * score[i]
    #     print(f"{species}: {confidence:.2f}%")


@app.route('/', methods=['GET'])
def check():
    print("Received.")
    data = {
        'message': 'Flask server is running properly.',
    }
    return jsonify(data)

@app.route('/api/data', methods=['GET'])
def get_data():
    data = {
        'message': 'Hello from Flask!',
        'data': [1,2,3,4,5]
    }
    return jsonify(data)

@app.route('/api/sendToModel', methods=['POST'])
def post_data():
    os.makedirs('./uploaded_files', exist_ok=True)
    file_path = None

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        file_path = f"./uploaded_files/{file.filename}"
        file.save(file_path)

        print(f"Received file: {file.filename} and saved to {file_path}")

        species, confidence = predict_species('uploaded_files/photo.jpg')
        confidence = round(confidence, 2)

        print(species)
        print(confidence)

        return jsonify({species: confidence}), 200
        # return jsonify({"message": "Successful"}), 200

    return jsonify({"error": "Unknown error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)