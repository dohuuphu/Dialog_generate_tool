import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from config.config import config

os.environ['KMP_DUPLICATE_LIB_OK']='True'
app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.environ["CUDA_VISIBLE_DEVICES"] = config["system"]["GPU_ID"]


from routers import face_recognition_api
app.include_router(
    router=face_recognition_api.router,
    tags=['face_recogntion']
)


if __name__ == "__main__":
    uvicorn.run("app:app", host=config["server"]["ip_address"], port=config["server"]["port"], reload=False)
    