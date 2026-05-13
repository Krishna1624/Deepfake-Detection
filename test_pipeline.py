import os
import sys
import numpy as np
import cv2

# Set path to current directory to import pipeline
sys.path.append(os.getcwd())

try:
    from pipeline import detect_deepfake
    print("Successfully imported detect_deepfake")
    
    video_path = "temp_video.mp4"
    if os.path.exists(video_path):
        print(f"Testing pipeline with {video_path}...")
        result = detect_deepfake(video_path)
        if "error" in result:
            print(f"PIPELINE ERROR: {result['error']}")
        else:
            print("PIPELINE SUCCESS!")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']}%")
            print(f"Frames Analyzed: {result['frames_analyzed']}")
            print(f"Raw Score: {result['raw_score']}")
            print(f"Processing Time: {result['processing_time']}s")
            print(f"Face Crops returned: {len(result['face_crops'])}")
            print(f"Frame Scores returned: {len(result['frame_scores'])}")
            print(f"Sample Frame Score: {result['frame_scores'][0]}")
    else:
        print(f"Video file {video_path} not found.")

except Exception as e:
    import traceback
    print(f"EXCEPTION DURING TEST: {e}")
    traceback.print_exc()
