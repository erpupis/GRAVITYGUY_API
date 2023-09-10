from entities.model_entity import Model
from core.database import Database
from utils.query_execution import execute_query
import base64

class ModelDao:

    def __init__(self, db: Database):
        self.db = db

    import base64

    def get_model_info(self, player_name: str) -> Model:
        query = '''
            SELECT player_name, train_start, train_end, parameters
            FROM MODELS
            WHERE player_name = %s 
        '''
        params = (player_name,)
        try:
            result = execute_query(query, params)
            if not result:
                return None

            # Extract fields from the tuple
            player_name, train_start, train_end, parameters = result[0]

            # Convert the 'parameters' field (BYTEA) to Base64
            parameters_base64 = base64.b64encode(parameters).decode("utf-8")

            return Model(player_name=player_name, train_start=train_start, train_end=train_end,
                         parameters=parameters_base64)
        except Exception as e:
            print(f"Error getting model info: {e}")
            raise

    def add_model(self, model: Model):
        fields = model.dict()  # Convert Pydantic model to dictionary
        columns = ', '.join(fields.keys())  # Columns for the query
        placeholders = ', '.join(['%s'] * len(fields))  # Placeholders for the query
        update_clause = ', '.join([f"{col} = EXCLUDED.{col}" for col in fields.keys()])

        query = f'''
            INSERT INTO MODELS ({columns}) VALUES ({placeholders})
            ON CONFLICT (player_name) DO UPDATE SET {update_clause}
        '''

        params = tuple(fields.values())  # Values for the query
        execute_query(query, params)

    def get_model(self, player_name: str) -> bytes:
        query = '''
            SELECT parameters
            FROM MODELS
            WHERE player_name = %s 
        '''
        params = (player_name,)
        try:
            result = execute_query(query, params)
            if not result:
                return None
            return result[0][0]
        except Exception as e:
            print(f"Error getting model: {e}")
            raise
