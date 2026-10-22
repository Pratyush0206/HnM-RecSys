FROM python:3.14-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY hmrec ./hmrec

ENV HMREC_MODEL=/data/ranker.txt
ENV HMREC_STORE=/data/store

EXPOSE 8000

CMD ["uvicorn", "hmrec.serve:app", "--host", "0.0.0.0", "--port", "8000"]