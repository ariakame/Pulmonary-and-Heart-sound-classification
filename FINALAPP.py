from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import librosa
import tensorflow as tf
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg'}

MODEL_CONFIG = {
    'heart': {
        'path': '/home/ariakame/Documents/IDP/cnn_melspec_model.h5',
        'classes': ['Abnormal', 'Normal'],
        'sample_rate': 4000,
        'duration': 4
    },
    'lung': {
        'path': '/home/ariakame/Documents/IDP/respiratory_gru_model.h5',
        'classes': [
            'Asthma', 'Bronchiectasis', 'Bronchiolitis', 'COPD',
            'Healthy', 'LRTI', 'Pneumonia', 'URTI'
        ],
        'sample_rate': 22050,
        'duration': 5
    }
}

heart_model = None
lung_model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():
    global heart_model, lung_model
    try:
        heart_model = tf.keras.models.load_model(MODEL_CONFIG['heart']['path'], compile=False)
        print("Heart model loaded.")
        lung_model = tf.keras.models.load_model(MODEL_CONFIG['lung']['path'], compile=False)
        print("Lung model loaded.")
    except Exception as e:
        print("Model loading error:", str(e))

def preprocess_audio(file_path, model_type):
    try:
        config = MODEL_CONFIG[model_type]
        sr = config['sample_rate']
        duration = config['duration']

        y, _ = librosa.load(file_path, sr=sr)
        max_len = sr * duration
        y = np.pad(y, (0, max(0, max_len - len(y))))[:max_len]

        if model_type == 'heart':
            n_mels = 128
            expected_width = 44
            hop_length = int((sr * duration) / expected_width)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
            if mel_spec_db.shape[1] != expected_width:
                mel_spec_db = mel_spec_db[:, :expected_width]
            return mel_spec_db.reshape((1, 128, expected_width, 1))

        else:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=52)
            features = np.mean(mfcc.T, axis=0)
            return features.reshape(1, 1, 52)

    except Exception as e:
        print("Preprocessing error:", str(e))
        return None

def predict_sound(file_path, model_type):
    try:
        model = heart_model if model_type == 'heart' else lung_model
        if model is None:
            return {'error': f'{model_type} model not loaded'}

        features = preprocess_audio(file_path, model_type)
        if features is None:
            return {'error': 'Failed to preprocess audio'}

        prediction = model.predict(features)
        classes = MODEL_CONFIG[model_type]['classes']

        prediction_flat = prediction.flatten()
        num_classes = min(len(classes), len(prediction_flat))
        predicted_class_idx = int(np.argmax(prediction_flat[:num_classes]))
        confidence = float(prediction_flat[predicted_class_idx])
        class_probs = {classes[i]: float(prediction_flat[i]) for i in range(num_classes)}

        # Diagnostic print to help debug confidence saturation
        print("Model type:", model_type)
        print("Feature mean:", np.mean(features))
        print("Feature std:", np.std(features))
        print("Prediction raw output:", prediction_flat)
        print("Predicted index:", predicted_class_idx)
        print("Confidence:", confidence)

        return {
            'predicted_class': classes[predicted_class_idx],
            'confidence': confidence,
            'probabilities': class_probs,
            'model_used': model_type
        }

    except Exception as e:
        return {'error': f'Prediction failed: {str(e)}'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'audio_file' not in request.files:
            return jsonify({'error': 'No file uploaded'})

        file = request.files['audio_file']
        model_type = request.form.get('model_type', 'heart')

        if file.filename == '':
            return jsonify({'error': 'No file selected'})

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type.'})

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
        file.save(file_path)

        result = predict_sound(file_path, model_type)
        os.remove(file_path)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'})

@app.route('/model_info')
def model_info():
    return jsonify({
        'heart': {
            'loaded': heart_model is not None,
            'classes': MODEL_CONFIG['heart']['classes']
        },
        'lung': {
            'loaded': lung_model is not None,
            'classes': MODEL_CONFIG['lung']['classes']
        }
    })

if __name__ == '__main__':
    load_models()
    os.makedirs('templates', exist_ok=True)
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w') as f:
            f.write('<h1>Heart & Lung Sound Analyzer</h1>')
    app.run(debug=True, host='0.0.0.0', port=8000)
