# Use a slim Python image
FROM python:3.10-slim

WORKDIR /app

# Install only the bare essentials for the database and compiler
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for any potential uploads
RUN mkdir -p uploads

# Start Gunicorn using Railway's dynamic port
# We use 'sh -c' to make sure the $PORT variable is read correctly at runtime
CMD ["sh", "-c", "gunicorn -w 4 -b 0.0.0.0:$PORT --timeout 120 --access-logfile - --error-logfile - backend_api:app"]
