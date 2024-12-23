# WhisperX API Server

A lightweight Flask API server that wraps WhisperX for efficient audio transcription using GPU acceleration. This service maintains a loaded model in memory for faster processing and provides a simple HTTP interface for transcription requests.

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