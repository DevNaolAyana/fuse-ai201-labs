# Sub-Module 6.3: Containerization with Docker

## What is Docker?

Docker packages the application and its entire environment (code, runtime, libraries, and configuration) into a single self-contained unit called a container. This ensures the API runs identically on a developer's laptop, a testing server, and a cloud platform (solving the "it works on my machine" problem).

## The Dockerfile

The `Dockerfile` in this directory defines the environment for our Credit Card Fraud Detection API:

1. **Base Image:** Uses `python:3.11-slim` to keep the container lightweight.
2. **Dependencies:** Copies `requirements.txt` and installs all necessary Python packages (FastAPI, XGBoost, etc.).
3. **Code:** Copies the `api/` and `models/` folders into the container.
4. **Execution:** Exposes port 8000 and runs the FastAPI app using `uvicorn`.

## How to build and run (if Docker is installed)

To build the image and run the container locally, the commands would be:

```bash
docker build -t fraud-api:v1.0 .
docker run -d -p 8000:8000 --name fraud-container fraud-api:v1.0
```
