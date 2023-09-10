from entities.input_entity import InputData
from core.database import Database
from utils.query_execution import execute_query
from datetime import datetime
import pandas as pd


class InputDao:
    def __init__(self, db: Database):
        self.db = db

    def add_input(self, input: InputData):
        fields = input.dict()  # Convert Pydantic model to dictionary
        columns = ', '.join(fields.keys())  # Columns for the query
        placeholders = ', '.join(['%s'] * len(fields))  # Placeholders for the query
        query = f'''
            INSERT INTO INPUTS ({columns}) VALUES ({placeholders})
            ON CONFLICT (player_name, run_start, fixed_frame) DO NOTHING
        '''
        params = tuple(fields.values())  # Values for the query
        execute_query(query, params)

    def delete_inputs_by_run(self, player_name: str, run_start: datetime):
        query = '''
            DELETE FROM INPUTS WHERE PLAYER_NAME = %s AND RUN_START = %s
        '''
        params = (player_name, run_start)
        execute_query(query, params)

    def get_inputs(self, player_name: str) -> InputData:
        query = '''
            SELECT RAYCAST_0, RAYCAST_0_HORIZONTAL, RAYCAST_30, RAYCAST_30_HORIZONTAL, RAYCAST_45, RAYCAST_45_HORIZONTAL, RAYCAST_315, RAYCAST_315_HORIZONTAL, RAYCAST_330, RAYCAST_330_HORIZONTAL, COLLECT_ANGLE, COLLECT_LENGTH, GRAVITY_DIR, SPEED, ON_GROUND_TOP, ON_GROUND_BOT, SWITCH_GRAVITY
            FROM INPUTS
            WHERE player_name = %s 
        '''
        params = (player_name,)
        try:
            result, columns = execute_query(query, params, col=True)
            if not result:
                return None
            return pd.DataFrame(result, columns=columns)
        except Exception as e:
            print(f"Error getting model: {e}")
            raise


