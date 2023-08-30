import psycopg2

from core.database import Database

db = Database()
db.initialize()
def execute_query(query, values=None, col=False):
    conn = db.get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, values)
            conn.commit()
            try:
                result = cursor.fetchall()
                if col:
                    columns = [desc[0] for desc in cursor.description]
                    return result, columns
                if not col:
                    return result
            except psycopg2.ProgrammingError:
                return None
    finally:
        db.release_connection(conn)
