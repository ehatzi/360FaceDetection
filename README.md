# 360FaceDetection
Face detection in 360 videos


Requirements - Specs
* Python 3.8
* Anaconda 
* PyTorch 1.9.1 CPU only: conda install pytorch torchvision torchaudio cpuonly -c pytorch
* mmcv 1.3.15: pip3 install mmcv (if you have CUDA: conda install -c esri mmcv-full)
* conda install ipython

1. MTCNN
  * facenet-pytorch: pip3 install facenet-pytorch (for mtcnn)

2. face_recognition instead of mtcnn (use Python 3.6 new env)
  * Installing dlib: conda install -c conda-forge dlib
  * pip install face-recognition (https://github.com/ageitgey/face_recognition) -> created dlib problems

3. SCRFD: https://github.com/deepinsight/insightface/tree/master/detection/scrfd
