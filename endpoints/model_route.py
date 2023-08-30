from fastapi import APIRouter, HTTPException
from entities.model_entity import Model
from daos.model_dao import ModelDao
from training.training import train

router1 = APIRouter()

def init_routes1(dao: ModelDao):

    @router1.get("/models/{player_name}", response_model=Model)
    async def get_model(player_name: str):
        try:
            model = dao.get(player_name)
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")
            return model
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router1.post("/model_train/")
    async def train_model(player_name: str):
        train_start, train_end, weights_base64, test_accuracy = train(player_name)
        model = Model(
            player_name=player_name,
            train_start=train_start,
            train_end=train_end,
            parameters=weights_base64
        )
        try:
            dao.add_model(model)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"message": "model added successfully"}, {"accuracy": test_accuracy*100}