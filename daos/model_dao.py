from entities.model_entity import Model
from core.database import Database
from utils.query_execution import execute_query
from datetime import datetime


class ModelDao:

    def __init__(self, db: Database):
        self.db = db

    def get(self, player_name: str) -> Model:
        query = '''
            SELECT player_name, train_start, train_end, parameters
            FROM MODELS
            WHERE player_name = %s 
        '''
        params = (player_name)
        try:
            result = execute_query(query, params)
            if not result:
                return None
            return Model(**result[0])
        except Exception as e:
            print(f"Error getting model: {e}")
            raise

    def add_model(self, model: Model):
        fields = model.dict()  # Convert Pydantic model to dictionary
        columns = ', '.join(fields.keys())  # Columns for the query
        placeholders = ', '.join(['%s'] * len(fields))  # Placeholders for the query
        query = f'''
                    INSERT INTO MODELS ({columns}) VALUES ({placeholders})
                '''
        params = tuple(fields.values())  # Values for the query
        execute_query(query, params)
