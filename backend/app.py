from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict as predict_func
from datetime import datetime
import json
import os

app = Flask(__name__, static_folder='../static')
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def safe_int(value):
    try:
        return int(value)
    except:
        return 0

@app.route('/predict', methods=['POST'])
def predict():
    # Validate intraoral image (optional now)
    if 'intraoral_image' in request.files and request.files['intraoral_image'].filename != '':
        intraoral_file = request.files['intraoral_image']
        if not allowed_file(intraoral_file.filename):
            return jsonify({'error': 'Invalid intraoral image type. Only PNG, JPG, JPEG allowed'}), 400
    else:
        intraoral_file = None
    
    # Validate histopath image (optional now)
    if 'histopath_image' in request.files and request.files['histopath_image'].filename != '':
        histopath_file = request.files['histopath_image']
        if not allowed_file(histopath_file.filename):
            return jsonify({'error': 'Invalid histopath image type. Only PNG, JPG, JPEG allowed'}), 400
    else:
        histopath_file = None
    
    # Safe type conversion with defaults
    age = safe_int(request.form.get("age", 0))
    gender = safe_int(request.form.get("gender", 0))
    tobacco = safe_int(request.form.get("tobacco", 0))
    smoking = safe_int(request.form.get("smoking", 0))
    alcohol = safe_int(request.form.get("alcohol", 0))
    family_history = safe_int(request.form.get("family_history", 0))
    oral_lesions = safe_int(request.form.get("oral_lesions", 0))
    unexplained_bleeding = safe_int(request.form.get("unexplained_bleeding", 0))
    difficulty_swallowing = safe_int(request.form.get("difficulty_swallowing", 0))
    patches = safe_int(request.form.get("patches", 0))
    
    try:
        result = predict_func(intraoral_file, histopath_file, {
            "age": age,
            "gender": gender,
            "smoking": smoking,
            "tobacco": tobacco,
            "alcohol": alcohol,
            "family_history": family_history,
            "oral_lesions": oral_lesions,
            "unexplained_bleeding": unexplained_bleeding,
            "difficulty_swallowing": difficulty_swallowing,
            "patches": patches
        })
        
        # Handle error tuple returns from predict_func
        if isinstance(result, tuple) and len(result) == 2:
            return jsonify(result[0]), result[1]
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def root():
    return jsonify({'message': 'Medical AI API is running'}), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.json if request.is_json else {}

        timestamp = datetime.now().isoformat()

        # Attach timestamp
        data["timestamp"] = timestamp

        # Attach predicted label safely (if present)
        if "result" in data:
            data["predicted_label"] = data["result"].get("final_risk")

        # Ensure feedback directory exists
        os.makedirs("backend/feedback_images", exist_ok=True)

        # Save feedback
        with open("backend/feedback.json", "a") as f:
            f.write(json.dumps(data) + "\n")

        return jsonify({"status": "success"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
