# import tensorflow as tf
# import os
# import json
# import numpy as np
# from tensorflow.keras.models import load_model
from flask import Flask, jsonify, request
from flask_cors import CORS

# model = None
app = Flask(__name__)
CORS(app)

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
    input_data = request

    print(input_data)
    
    return jsonify({"message": "Data posted successfully"}), 200

if __name__ == '__main__':
    
    # model_path = 'v7_flower.keras'
    # img_dimensions = 240
    # img_height, img_width =  img_dimensions, img_dimensions  # image dimensions

    # if os.path.isfile(model_path):
    #     print("File exists")
    #     model = load_model(model_path)
    # else:
    #     print("File not found")
        
    # model.summary()
    # model.compile(optimizer='adam',
    #             loss='categorical_crossentropy',
    #             metrics=['accuracy'])
    
    app.run(debug=True, host='192.168.1.49', port=5000)