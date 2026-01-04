FROM python:3.11-slim

# ==============================
# Environment settings
# ==============================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ==============================
# Working directory
# ==============================
WORKDIR /app

# ==============================
# System dependencies (XGBoost)
# ==============================
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ==============================
# Python dependencies
# ==============================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ==============================
# Copy only required files
# ==============================
COPY src ./src
COPY models ./models
COPY templates ./templates

# ==============================
# Expose API port
# ==============================
EXPOSE 8000

# ==============================
# Run FastAPI
# ==============================
CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000"]
