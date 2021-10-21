
import face_recognition
#import torch
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw
from IPython import display



filename = 'C:/Users/ehatzi/Documents/Projects/MediaVerse/VRAG_videos/SAM_100_0026'

video = mmcv.VideoReader(filename + '.mp4')
#frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]
frames_tracked = []
for i, frame in enumerate(video):
    print('\rTracking frame: {}'.format(i + 1), end='')
    
    # Detect faces
    boxes = face_recognition.face_locations(frame)
    
    # Draw faces
    frame_draw = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).copy()
    draw = ImageDraw.Draw(frame_draw)
    if boxes is not None:
        for box in boxes:
            #draw.rectangle(box, outline=(255, 0, 0), width=6)
            box2 = list(box)
            draw.rectangle([box2[3],box2[0],box2[1],box2[2]], outline=(255, 0, 0), width=6, fill="yellow")
    
    # Add to frame list
    #frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))
    frames_tracked.append(frame_draw)
print('\nDone')

dim = frames_tracked[0].size
fourcc = cv2.VideoWriter_fourcc(*'mp4v')    
video_tracked = cv2.VideoWriter(filename + '_tracked_facerec.mp4', fourcc, 25.0, dim)
for frame in frames_tracked: 
    video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
video_tracked.release()