# Base Python image
FROM python:3.10

# Set workdir
WORKDIR /app

# Copy everything
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirenmnets.txt

# Expose port for FastAPI
EXPOSE 8000

# Start FastAPI app
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
