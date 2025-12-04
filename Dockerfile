FROM python:3.11-slim-bullseye

WORKDIR /application
COPY . /application

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends apt-utils awscli \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "application.py"]
