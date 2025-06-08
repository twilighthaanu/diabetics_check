# Use Python 3.9 slim image
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p data/uploads

# Expose ports for Flask and Streamlit
EXPOSE 5000 8501

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=src/api/app.py
ENV FLASK_ENV=development

# Command to run both services
CMD ["sh", "-c", "streamlit run src/frontend/app.py & python src/api/app.py"] 