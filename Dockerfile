FROM python:3.11-slim

# 1. Set working directory
WORKDIR /app

# 2. Python runtime settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. System dependencies (add more here if your libs need them)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# 4. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the repo
COPY . .

# 6. Streamlit server config
ENV PORT=8501
EXPOSE 8501

# 7. Launch the same panel as you run locally
CMD ["streamlit", "run", "src/ui/rag_panel.py", "--server.port=8501", "--server.address=0.0.0.0"]
