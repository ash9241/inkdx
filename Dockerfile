FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml LICENSE README.md ./
COPY src ./src
COPY calibration ./calibration

RUN pip install --no-cache-dir .

ENTRYPOINT ["inkdx"]
CMD ["--help"]
