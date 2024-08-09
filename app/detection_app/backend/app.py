from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/process', methods=['POST'])
def process_data():
    data = request.json

    # this is where you would process the data 
    processed_data = {
        "received": data,
        "additional_info": "This is some additional data"
    }

    return jsonify(processed_data)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)