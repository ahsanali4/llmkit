from typing import Optional

from pydantic import BaseSettings, Field, SecretStr


class Config(BaseSettings):
    aa_api_host: Optional[str] = Field(env="API_HOST")
    openai_api_key: Optional[SecretStr] = Field(env="openai_api_key")
    aa_token: Optional[SecretStr] = Field(env="AA_TOKEN")
    aleph_alpha_api_key: Optional[SecretStr] = Field(env="ALEPH_ALPHA_API_KEY")

    class Config:
        env_prefix = ""
        case_sentive = False
        env_file = ".env"
        env_file_encoding = "utf-8"
