FROM python:3.12-slim

WORKDIR /app

# Update and install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . /app
RUN chmod -R 777 /app

# Set writable cache paths
ENV STREAMLIT_HOME="/.streamlit"
ENV TORCH_HOME="/.cache/torch"
ENV GDOWN_HOME="/.cache/gdown"
ENV XDG_CACHE_HOME="/.cache"

# Create cache/config directories
RUN mkdir -p $STREAMLIT_HOME $TORCH_HOME $GDOWN_HOME $XDG_CACHE_HOME /app/assets \
    && chmod -R 777 $STREAMLIT_HOME $TORCH_HOME $GDOWN_HOME $XDG_CACHE_HOME /app/assets


ENV STREAMLIT_HOME="/app/.streamlit"
ENV TORCH_HOME="/app/.cache/torch"
ENV GDOWN_HOME="/app/.cache/gdown"
ENV XDG_CACHE_HOME="/app/.cache"

RUN mkdir -p $STREAMLIT_HOME $TORCH_HOME $GDOWN_HOME $XDG_CACHE_HOME /app/assets \
    && chmod -R 777 $STREAMLIT_HOME $TORCH_HOME $GDOWN_HOME $XDG_CACHE_HOME /app/assets

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Disable CORS (use only if safe)
ENV STREAMLIT_SERVER_ENABLECORS=false

# Health check for Docker
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
ENV PYTHONUNBUFFERED=1
# Run Streamlit

ENTRYPOINT ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]

