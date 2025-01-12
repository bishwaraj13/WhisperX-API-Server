# WhisperX API Server

A lightweight Flask API server that wraps WhisperX for efficient audio transcription using GPU acceleration. This service maintains a loaded model in memory for faster processing and provides a simple HTTP interface for transcription requests.

## Context
WhisperX has very specific dependencies which run on older python and torch versions, and is usually incompatible with some of the newer applications, so I separated out WhisperX, and made it a simple Flask API server. That way my code in other applications can just send API request to this server, and receive transcription as response.

## Features

- **Persistent Model Loading**: Keeps WhisperX model loaded in memory for faster processing
- **GPU Acceleration**: Utilizes CUDA for efficient transcription
- **RESTful API**: Simple HTTP interface for transcription requests
- **Health Monitoring**: Endpoint to check service status and GPU memory usage

## Prerequisites

- Docker
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit
- Audio files accessible from the host machine

## Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/whisperx-api-server.git
cd whisperx-api-server
```

2. **Build the Docker image**
```bash
docker build -t whisperx-api .
```

3. **Run the container**
```bash
docker run --gpus all --shm-size=8gb --runtime=nvidia --network=host --volume="/sizzle_storage:/sizzle_storage:rw" --volume="$(pwd):/app" -p 5000:5000 --name whisperx-api whisperx-api
```

4. **Wait
After running the container, it takes about 5 minutes while for Whisper model to download and load. Wait till these messages appears in screen:

```bash
 * Serving Flask app 'app'
 * Debug mode: off
INFO:werkzeug:WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.3.193:5000
INFO:werkzeug:Press CTRL+C to quit
```

Once the above message appears, you are good to start making requests for API calls.

You will also be able to see a process called `WhisperX-API` loaded in GPU memory once you run the command `nvidia-smi`.

## API Endpoints
POST `/transcribe`
```json
{
    "file_path": "/audio/your-audio-file.wav"
}
```

## Health Check
GET `/health`
```json
{
    "status": "healthy",
    "model_loaded": true,
    "gpu_memory_allocated": 4521.5,
    "gpu_memory_cached": 6144.0
}
```

## Usage Examples using Python
```python
import requests

def transcribe_audio(file_path):
    url = "http://localhost:5000/transcribe"
    payload = {"file_path": file_path}
    response = requests.post(url, json=payload)
    return response.json()

# Example usage
result = transcribe_audio("/audio/sample.wav")
print(result["segments"])
```

## Directory structure
```javascript
whisperx-api-server/
├── Dockerfile
├── README.md
├── app.py
```

## Bibliography

[Whisper](https://github.com/m-bain/whisperX)

[Flask](https://github.com/pallets/flask)
