from fastapi import APIRouter, HTTPException
from daos.runs_dao import RunDao
from datetime import datetime
from deserialization.deserialization import extract_run, deserialize
from daos.input_dao import InputDao
from os import listdir
from os.path import isfile, join, splitext

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

    @router3.delete("/runs_delete/")
    async def delete_run(player_name: str, run_start: datetime):
        try:
            dao.delete_run(player_name, run_start)
            dao1.delete_inputs_by_run(player_name, run_start)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"message": "Run deleted successfully"}

    @router3.post("/runs-dir/")
    async def add_runs_fromdir(directory_path: str):
        try:
            # Get all files in the directory
            file_names = [f for f in listdir(directory_path) if isfile(join(directory_path, f))]

            # Create sets for json and dat files
            json_files = {splitext(f)[0]: f for f in file_names if f.endswith('.json')}
            dat_files = {splitext(f)[0]: f for f in file_names if f.endswith('.dat')}

            # Find common base names and process them
            common_bases = set(json_files.keys()) & set(dat_files.keys())
            for base in common_bases:
                json_path = join(directory_path, json_files[base])
                dat_path = join(directory_path, dat_files[base])

                run = extract_run(json_path)
                inputs = deserialize(json_path, dat_path)

                dao.add_run(run)
                for input in inputs:
                    dao1.add_input(input)

        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        return {"message": f"{len(common_bases)} runs added successfully"}