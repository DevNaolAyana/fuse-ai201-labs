# Cloud Deployment Plan (Render)

## Platform Comparison

- **Render:** Free tier, simple GitHub integration, supports Docker. Good for demos (but cold starts take ~50 seconds).
- **Hugging Face Spaces:** Free, great for ML demos, but requires modifying the Dockerfile to run on their specific ports.
- **Decision:** Render is the best choice because it natively builds our `Dockerfile` and allows environment variable injection via the dashboard.

## Deployment Steps (If executed on Render)

1. Push the project to a GitHub repository.
2. Log into Render.com and create a new "Web Service".
3. Connect the GitHub repository. Set Runtime to "Docker".
4. Under Environment Variables, add `MODEL_PATH`, `SCALER_PATH`, and `THRESHOLD` (values from `.env.example`).
5. Set the Health Check Path to `/health`.
6. Deploy. Render will build the Docker image and provide a public URL.

## Configuration & Secrets Evidence

- Secrets are **not** hardcoded in the code.
- The `.gitignore` excludes `.env`.
- The `.dockerignore` excludes `.env` so it doesn't get baked into the Docker image.
- `api/config.py` uses `pydantic-settings` to load variables from the environment.

## Cold-Start Management

On Render's free tier, the service sleeps after 15 minutes of inactivity. The first request wakes it up, taking ~50 seconds.
**Plan:** 5 minutes before a live demo or video presentation, I will send a dummy request to the `/health` endpoint to "wake up" the server, ensuring the live prediction is instant.

## Rollback Plan

1. **Last known good version:** Tag `v1.0` (commit `b09be70`).
2. **How to redeploy:** In the Render dashboard, select the `v1.0` commit from the deploy history and click "Redeploy".
3. **Who:** Naol Melaku Ayana (Project Lead).
4. **User communication:** If the rollback happens during the capstone, I will notify the instructor and group via the Telegram group chat.

## Statement of Deployment Status

_Note: Due to local storage and account constraints, the Docker container was tested locally using `uvicorn`, but the public cloud deployment to Render was not executed. This document serves as the complete, professional deployment plan._
