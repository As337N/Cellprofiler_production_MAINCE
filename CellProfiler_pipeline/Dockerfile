FROM python:3.8-bullseye

RUN sed -i 's|deb.debian.org|archive.debian.org|g; /bullseye-security/d' /etc/apt/sources.list

RUN apt-get -o Acquire::Check-Valid-Until=false update && apt-get install -y \
    openjdk-11-jdk-headless \
    build-essential \
    libgtk-3-dev \
    bc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /tmp/matplotlib
ENV MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_DISABLE_JIT=0 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    PYTHONUNBUFFERED=1 \
    JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

RUN apt-get install -y fonts-dejavu-core

CMD ["python", "-u", "main.py"]
