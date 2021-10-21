from facenet_pytorch import MTCNN
#import face_recognition
import torch
import numpy as np
import mmcv, cv2
from PIL import Image, ImageDraw
from IPython import display

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

filename = 'C:/Users/ehatzi/Documents/Projects/MediaVerse/VRAG_videos/SAM_100_0026'

mtcnn = MTCNN(keep_all=True, device=device)
video = mmcv.VideoReader(filename + '.mp4')
frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]
frames_tracked = []
for i, frame in enumerate(frames):
    print('\rTracking frame: {}'.format(i + 1), end='')
    
    # Detect faces
    boxes, _ = mtcnn.detect(frame)
    
    # Draw faces
    frame_draw = frame.copy()
    draw = ImageDraw.Draw(frame_draw)
    if boxes is not None:
        for box in boxes:
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
    
    # Add to frame list
    #frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))
    frames_tracked.append(frame_draw)
print('\nDone')

dim = frames_tracked[0].size
fourcc = cv2.VideoWriter_fourcc(*'mp4v')    
video_tracked = cv2.VideoWriter(filename + '_tracked_mtcnn.mp4', fourcc, 25.0, dim)
for frame in frames_tracked: 
    video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
video_tracked.release()