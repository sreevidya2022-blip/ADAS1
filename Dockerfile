# Use a slim Python image to keep the build light 
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
# Added libgl1 and libglib2.0-0 for OpenCV (cv2) compatibility 
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies 
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code 
COPY . .

# Create necessary directories 
RUN mkdir -p uploads

# Expose the default port (Railway will override this with $PORT) 
EXPOSE 5000

# Start Gunicorn using a shell to ensure $PORT is correctly evaluated 
CMD ["sh", "-c", "gunicorn -w 4 -b 0.0.0.0:$PORT --timeout 120 --access-logfile - --error-logfile - backend_api:app"]
