# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies if needed (e.g., for certain libraries)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# Install Poetry (or directly use pip with requirements.txt)
# Using pip for simplicity based on initial setup
COPY requirements.txt .
# Assuming requirements.txt exists and lists FastAPI, Uvicorn, Polars, Redis, etc.
# RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY ./app /app/app
COPY ./data/processed /app/data/processed
# Ensure the models directory exists within the app directory in the container
RUN mkdir -p /app/app/models
COPY ./app/models /app/app/models

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the application
# Use the app.main:app entry point
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
