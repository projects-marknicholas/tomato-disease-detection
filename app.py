from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import base64

app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})

# Load YOLOv8 model
model = YOLO("best.pt")

# Define a mapping from class ID to disease name
CLASS_ID_TO_NAME = {
    0: "Anthrocnose",
    1: "Bacterial_Spot",
    2: "Botrytis",
    3: "fruitworm_bore",
    4: "tomato_cracking"
}

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    image = Image.open(file.stream)

    # Perform detection using YOLOv8
    results = model(image)

    # Convert results to a JSON serializable format
    detections = []
    for result in results[0].boxes.data:
        class_id = int(result[5].item())
        detections.append({
            'class': class_id,  # Class ID
            'name': CLASS_ID_TO_NAME.get(class_id, "Unknown"),  # Disease name
            'confidence': float(result[4].item()),  # Confidence score
            'box': [float(coord) for coord in result[:4].tolist()]  # Bounding box
        })

    # Convert the image to a base64 string for display
    output_image_stream = io.BytesIO()
    image.save(output_image_stream, format='PNG')
    output_image_stream.seek(0)
    image_base64 = base64.b64encode(output_image_stream.getvalue()).decode('utf-8')

    return jsonify({'detections': detections, 'image_url': 'data:image/png;base64,' + image_base64})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
