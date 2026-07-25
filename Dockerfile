# Use an official Python image as a base
FROM python:3.11-slim as python-base

# Set working directory
WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g pnpm

# Copy requirements and install Python dependencies
COPY ML-Backend/src/requirements.txt ./ML-Backend/src/
RUN pip install --no-cache-dir -r ML-Backend/src/requirements.txt

# Copy backend dependencies
COPY backend/package*.json ./backend/
WORKDIR /app/backend
RUN npm install --production

# Copy the rest of the application
WORKDIR /app
COPY . .

# Create uploads directory
RUN mkdir -p backend/uploads

# Set environment variables
ENV PORT=8000
ENV NODE_ENV=production
ENV PYTHON_BIN=python3

# Expose the port
EXPOSE 8000

# Create a non-root user for Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# Start the application
WORKDIR /app/ML-Backend
CMD ["python3", "app.py"]
