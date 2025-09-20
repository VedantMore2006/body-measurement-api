FROM python:3.12-slim

# Set workdir
WORKDIR /app

# Copy everything
COPY . .

# Install CPU-only Torch & Torchvision first (skips CUDA/NVIDIA deps)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Then install your requirements (ultralytics will use the CPU Torch)
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Cloud Run loves
EXPOSE 8080

# Run with uvicorn, grabbing PORT env
CMD ["python", "main.py"]