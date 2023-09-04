import pydantic
from datetime import datetime

class InputData(pydantic.BaseModel):
    player_name: str
    run_start: datetime
    fixed_frame: int
    raycast_0: float
    raycast_30: float
    raycast_45: float
    raycast_315: float
    raycast_330: float
    collect_angle: float
    collect_length: float
    gravity_dir: float
    speed: float
    on_ground_top: bool
    on_ground_bot: bool
    switch_gravity: bool
