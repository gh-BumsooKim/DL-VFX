# DeepLearning VFX

**중앙대학교 첨단영상대학원 영상학과 - 딥러닝 시각효과**

**Face Generative Model**
<br>[Bumsoo Kim](https://github.com/gh-BumsooKim)\* (\*Graphics Realization Lab, CAU)
<br>In Deep Learning VFX Lecture, the graduate school of Advanced Imaging Science, Multimedia & Film, Chung-Ang University


## Application

| Index | Model | Epochs | Result | Generated Image | Note❗ | Visualizing | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | [Simple Generative Model V1](face_generate_model_v1_failure.ipynb) | 500 | **Failure** | ![](imgs/v1_1_to_7_epoch.gif) | Mode collapse after 8 Epoch | [v1_vis.ipynb](face_generate_model_v1_failure_vis.ipynb) |
| 2 | [Simple Generative Model V2](face_generate_model_v2_failure.ipynb) | 120 | **Failure** | ![](imgs/v2_2_to_12_epoch.gif) | Mode collapse after 13 Epoch | [v2_vis.ipynb](face_generate_model_v2_failure_vis.ipynb) |

## Environment
- WIndows10 (OS)
- Python 3.7.9
- CUDA 11.1
- cuDNN 8.0.5
- Pytorch : 1.8.1 with CUDA 11.1
- GeForce GTX 3080 10GB (GPU) / Compute Capability 8.6
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

- Anaconda3 :
```cmd
(base)conda create -n venv python=3.7.8
(base)conda activate venv

(venv)conda install -n venv spyder
```

### (check your environment)
- CUDA :
```cmd
nvcc -v # nvcc --version
which cuda
```

- cuDNN :
```cmd
cat /usr/lib/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2 # your cudnn.h PATH
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
