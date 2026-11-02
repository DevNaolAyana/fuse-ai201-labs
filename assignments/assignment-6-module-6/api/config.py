from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_path: str = "models/fraud_detector_model.pkl"
    scaler_path: str = "models/feature_scaler.pkl"
    threshold: float = 0.5

    class Config:
        env_file = ".env"

def get_settings():
    return Settings()