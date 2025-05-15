from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = True
    allowed_origins: list[str] = ["*"]

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()
