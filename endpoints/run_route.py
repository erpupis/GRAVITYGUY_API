from fastapi import APIRouter, HTTPException
from entities.run_entity import Run
from daos.runs_dao import RunDao
from datetime import datetime
from deserialization.deserialization import extract_run, deserialize
from daos.input_dao import InputDao

router3 = APIRouter()

def init_routes3(dao: RunDao, dao1: InputDao):

    @router3.post("/runs/")
    async def add_run(json_path: str, dat_path: str):
        run = extract_run(json_path)
        inputs = deserialize(json_path, dat_path)
        try:
            dao.add_run(run)
            for input in inputs:
                dao1.add_input(input)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"message": "Run added successfully"}


    @router3.post("/runs/")
    async def add_run(run: Run):
        try:
            dao.add_run(run)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"message": "Run added successfully"}

    @router3.delete("/runs/")
    async def delete_run(player_name: str, run_start: datetime):
        try:
            dao.delete_run(player_name, run_start)
            dao1.delete_inputs_by_run(player_name, run_start)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"message": "Run deleted successfully"}