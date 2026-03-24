# 1. Use a lightweight Python image
FROM python:3.12-slim

# 2. Install system dependencies for OpenCV & YOLO
# 'headless' environments need these libraries to process images
# RUN apt-get update && apt-get install -y \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/*

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the entire project structure
# This ensures src/ui/app.py can find ../config.yaml
COPY . .

# 6. Expose the port Streamlit uses
EXPOSE 8501

# 7. Command to run the app
# We run it from the root so the relative paths in your code stay valid
CMD ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]