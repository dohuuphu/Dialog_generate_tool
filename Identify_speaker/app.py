import sys
sys.path.append('.')
import time

from config.config import  get_config
import socketio
import uvicorn

from fastapi import FastAPI
from concurrent.futures.thread import ThreadPoolExecutor
from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles

from modules.consumers import consumers

from api.route import SpksVerify_setup_route

from inference import Model

def set_up_app():
    app = FastAPI(name='spks_verify')
    origins = ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app



if __name__ == '__main__':
    config_app = get_config()
    app = set_up_app()
    sio = socketio.AsyncServer(cors_allowed_origins='*', async_mode='asgi')
    verify_model = Model(config_app)
    
    # start = time.time()
    # # model.save_newEmb('phu', '/AIHCM/ASR/phudo/SpeechVerify/audio/c.wav')
    # print(verify_model.verify_speakers('/AIHCM/ASR/phudo/SpeechVerify/audio/H_cao.wav'))
    # print("infer", time.time()- start)
    SpksVerify_setup_route(app, config_app, verify_model)
    
    socketio_app  = socketio.ASGIApp(sio, app)

    consumers(sio ,verify_model)
    
    uvicorn.run(socketio_app ,host='0.0.0.0', port=config_app['deployment']['port'])