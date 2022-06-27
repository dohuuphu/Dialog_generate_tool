from typing import List
from fastapi import APIRouter, HTTPException, UploadFile, File, Body
import glob
from config.config import config
from process.storage_handler import get_data_kvqa_from_storage, get_image_from_storage, get_subject_image_from_storage
from process.inference import inference_api
import time
router = APIRouter()
# data_kvqa = get_data_kvqa_from_storage(config["storage"]["kvqa"]["path"])
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio

executor = ThreadPoolExecutor(5)

async def join(url):
    print(0)
    def join_thread(url):
        return  inference_api(url)
    
    return await asyncio.get_event_loop().run_in_executor(executor, join_thread, url)

@router.get("/KVQA")
async def create_upload_file(url: str):
    await join(url)
    # if isinstance(result, str):
    #     raise HTTPException(status_code=404, detail=result)
    
    # result["id"] = len(data_kvqa)
    # data_kvqa.append(result)
    # return result

@router.get("/attendance_information/{name}")
async def get_data(name: str):
    attendance_information = {
        "name" : name,
        "date": []
    }
    information = glob.glob("./libs/result/*.jpg")
    for data in information:
        data = data.replace(".jpg","")
        data = data.split("/")[-1]
        print(data)
        data_name = data.split("_")[0]
        if name in data_name:
            attendance_information["date"].append("_".join(data.split("_")[1:]))
    return attendance_information



# @router.get("/get_object_info/{image_id}/{object_id}")
# async def get_data(image_id: str, object_id:str):
#     for data in data_kvqa:
#         if data["image_id"] == image_id:
#             image_information = get_subject_image_from_storage(config["storage"]["kvqa"]["path"], image_id, object_id)
#             return image_information

#     raise HTTPException(status_code=404, detail="Không tìm thấy ảnh")

