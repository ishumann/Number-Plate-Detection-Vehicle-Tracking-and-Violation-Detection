# Use Python 3.8 slim image as base
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data models

# Download YOLOv5 weights (you may need to modify this based on your specific model)
RUN python -c "import torch; torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt', 'models/yolov5s.pt')"

# Expose port for Streamlit
EXPOSE 8501

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Create entrypoint script
RUN echo '#!/bin/bash\n\
cd /app/src/ui\n\
streamlit run app.py &\n\
cd /app/src\n\
python main.py\n\
wait' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"] 