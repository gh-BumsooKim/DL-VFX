# DeepLearning VFX

중앙대학교 첨단영상대학원 영상학과 - 딥러닝 시각효과 강의

## Environment
- Ubuntu 20.04 LTS (OS)
- Python 3.7.8
- CUDA 10.1
- cuDNN 7.6.5
- GeForce GTX 1060 3GB (GPU) / Compute Capability 6.1
- Install Package :
```cmd
sudo pip3 install -r requirements.txt
```

### (for python virtual environment user)
- Python3 venv (not Python2 virtualenv, and pyvenv) :
```cmd
cd [dir_path]
python -m venv .venv
source .venv/bin/activate
```
### (check your environment)
- CUDA :
```cmd
nvcc -v #nvcc --version
whereis cuda
```

- cuDNN :
```cmd
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2 # your cudnn.h PAT
```

- GPU driver :
```cmd
nvidia-smi
```

### (check nvidia GPU connection)
- TensorFlow 2.X:
```python
import tensorflow as tf
tf.config.list_physical_devices("GPU")
tf.test.is_gpu_available()
```
- PyTorch :
```python
import torch
torch.cuda.device_count()
torch.cuda.get_device_name(0)
torch.cuda.is_available()
```
