# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all project files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Hugging Face default port
EXPOSE 7860

# Run Streamlit when container starts
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
