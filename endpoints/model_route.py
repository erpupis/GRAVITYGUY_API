from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO
from entities.model_entity import Model
from daos.model_dao import ModelDao
from training.training import train

router1 = APIRouter()

def init_routes1(dao: ModelDao):

    @router1.get("/model_info/{player_name}", response_model=Model)
    async def get_model_info(player_name: str):
        try:
            model = dao.get_model_info(player_name)
            if not model:
                raise HTTPException(status_code=404, detail="Model Info not found")
            return model
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router1.get("/model/{player_name}")
    async def get_model(player_name: str):
        try:
            model = dao.get_model(player_name)
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")
            # Create a BytesIO stream from the model bytes and use it for the StreamingResponse
            return StreamingResponse(BytesIO(model[0]), media_type="application/octet-stream")
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router1.get("/download_model/{player_name}", response_class=StreamingResponse)
    def download_model(player_name: str):
        model_data = dao.get_model(player_name)

        if model_data is None:
            raise HTTPException(status_code=404, detail="Model not found")

        # Create a BytesIO object from your model_data
        return StreamingResponse(BytesIO(model_data), media_type="application/octet-stream",
                                 headers={"Content-Disposition": f"attachment;filename={player_name}.onnx"})


    @router1.post("/model_train/")
    async def train_model(player_name: str):
        train_start, train_end, onnx_model, nn_accuracy, nn_precision = train(player_name)
        model = Model(
            player_name=player_name,
            train_start=train_start,
            train_end=train_end,
            parameters=onnx_model
        )
        try:
            dao.add_model(model)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"message": "model added successfully"}, {"accuracy": nn_accuracy*100}, {"precision": nn_precision*100}