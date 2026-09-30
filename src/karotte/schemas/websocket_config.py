from pydantic import BaseModel


class WebSocketConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8001
