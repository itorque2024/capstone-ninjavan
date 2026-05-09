FROM python:3.10-slim

WORKDIR /app

# System deps: libgomp1 required by LightGBM + Prophet on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Set environment variables for Hugging Face Spaces
ENV HOST=0.0.0.0
ENV PORT=7860

# Expose port
EXPOSE 7860

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "app.main_api:app", "--host", "0.0.0.0", "--port", "7860"]
