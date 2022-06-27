import time
from os.path import dirname, abspath, join
from fastapi import File, UploadFile,status,Form
from variables import *

from api.response import APIResponse
from api.util import save_file

def SpksVerify_setup_route(app, appConfig, verify_model):
    
    @app.post('/verify_speakers')
    async def verify_speakers(_file: UploadFile):
        
        if not _file.filename.endswith(('.mp3', '.flac', '.wav', '.aac', '.m4a', '.weba', '.sdt','.mp4')):
            return APIResponse.json_format(False,'File type is not supported',status.HTTP_400_BAD_REQUEST)
    
        file_path = join(dirname(dirname(abspath(__file__))), appConfig["upload"], _file.filename )
        
        st = time.time()
        stt, score, speaker = verify_model.verify_speakers(await save_file(_file, file_path))
        time_process = time.time() - st

        return APIResponse.json_format(True, 'Success', status.HTTP_200_OK, stt, time_process, score, speaker)

    @app.post('/add_speakers')
    async def add_speakers( _file: UploadFile, name_speaker: str = Form('')):
        if not _file.filename.endswith(('.mp3', '.flac', '.wav', '.aac', '.m4a', '.weba', '.sdt','.mp4')):
            return APIResponse.json_format(False,'File type is not supported',status.HTTP_400_BAD_REQUEST)

        file_path = join(dirname(dirname(abspath(__file__))), appConfig["upload"], _file.filename )
        print('name_speaker', name_speaker)
        result = verify_model.save_newEmb(name_speaker , await save_file(_file, file_path))