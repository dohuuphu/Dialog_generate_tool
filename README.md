# Dialog generator tool
This tool is intended to create dialog data

# Quickstart
You can install the Python dependencies with
```
pip3 install -r requirements.txt
```

## Inference

```
python inference.py -p 'video_path' -c True
```
-p: video path (str)
-c: cross check by Computer Vision (bool)
- The result folder (ASR_result and CV_result) will be created at the folder that contain input_video
- The output_file is located at ASR_result folder as txt file
