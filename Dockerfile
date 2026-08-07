FROM python:3.10-slim

# Set useful defaults such as: code lives in /app, generated data in /data, logs in /logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    JUNO_BASE_DIR=/app \
    JUNO_DATA_DIR=/data \
    JUNO_LOG_DIR=/logs \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

#install system packages. The eddies script requires nco, which is not available in the python:3.10-slim image
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        cron \
        nco \
        ca-certificates \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

#copy and install python dependencies from the requirements.txt file
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt

#copy the source code and DOcker runtime files to /app
COPY src/ /app/src/
COPY docker/ /app/docker/
COPY README.md LICENSE.txt /app/

#make the entrypoint.sh file executable and create the /data and /logs directories
RUN chmod +x /app/docker/entrypoint.sh && mkdir -p /data /logs

#make the /data and /logs directories available as volumes for persistent storage
VOLUME ["/data", "/logs"]

#container runs the entrypoint.sh by default and unless specified otherwise, runs the cron daemon to execute the scheduled tasks
ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["cron"]
