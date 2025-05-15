# Use official Python image as base
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (to leverage Docker cache)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose port 8000 (FastAPI default)
EXPOSE 8000

# Command to run the app with reload for dev; remove --reload in prod
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
