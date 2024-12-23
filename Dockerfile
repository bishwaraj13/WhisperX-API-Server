# Use NVIDIA CUDA base image with development libraries
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Download and install additional CUDA libraries
RUN wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run && \
    sh cuda_11.8.0_520.61.05_linux.run --toolkit --silent && \
    rm cuda_11.8.0_520.61.05_linux.run

# Set CUDA environment variables
ENV PATH=/usr/local/cuda-11.8/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH}

# Install numpy with version constraint
RUN pip3 install 'numpy<2.0'

# Install PyTorch and other dependencies
RUN pip3 install torch==2.0.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118

# Install WhisperX
RUN pip3 install git+https://github.com/m-bain/whisperx.git

RUN pip3 install flask

# Expose the port
EXPOSE 5000

# Run the Flask app
CMD ["python3", "app.py"]
