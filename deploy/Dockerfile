FROM python:3.13-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PORT=12000 HOST=0.0.0.0

# lightgbm needs libgomp; keep the image small otherwise.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 12000
# The script re-execs the app on exit; Docker's restart policy covers crashes.
CMD ["./scripts/run_server.sh"]
