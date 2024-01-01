FROM huggingface/transformers-pytorch-gpu

# Install libraries
COPY ./requirements.txt ./
RUN pip install -r requirements.txt && rm ./requirements.txt

# Setup container directories
RUN mkdir /app

# Copy local code to the container
COPY ./app /app
WORKDIR /app

# (Debugging) See files in current folder.
RUN ls

# Download models (into container?)
RUN python3 download_models.py

EXPOSE 8080
CMD ["gunicorn", "main:app", "--timeout=0", "--preload", "--workers=1", "--threads=4", "--bind=0.0.0.0:8080"]
