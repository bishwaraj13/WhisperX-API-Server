from flask import Flask, request, jsonify
import whisperx
import torch
import setproctitle
import gc
import os

# Set process title at the very beginning of your app
setproctitle.setproctitle("WhisperX-API")

app = Flask(__name__)

# Global variables to store models
whisper_model = None
device = "cuda"
compute_type = "float16"
batch_size = 64

def initialize_model():
    global whisper_model
    if whisper_model is None:
        whisper_model = whisperx.load_model("large-v2", device, compute_type=compute_type)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    global whisper_model
    
    data = request.get_json()
    if 'file_path' not in data:
        return jsonify({'error': 'No file path provided'}), 400
        
    file_path = data['file_path']
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # Use the already loaded model
        audio = whisperx.load_audio(file_path)
        result = whisper_model.transcribe(audio, batch_size=batch_size)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add a health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': whisper_model is not None,
        'gpu_memory_allocated': float(torch.cuda.memory_allocated()) / 1024**2,  # Convert to MB
        'gpu_memory_cached': float(torch.cuda.memory_cached()) / 1024**2  # Convert to MB
    })

if __name__ == '__main__':
    # Initialize model before starting the server
    initialize_model()
    app.run(host='0.0.0.0', port=5000)