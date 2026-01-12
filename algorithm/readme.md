## Environment Setup (Windows / Conda + pip)

### 1) Create conda env (binary deps via conda)
```bash
conda create -n pinn python=3.9 -y
conda activate pinn

# Install PyTorch stack from conda (avoid pip mixing!)
conda install -y pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
````

### 2) Install python deps via pip (pure python deps)

```bash
pip install -r requirements.txt
```

### 3) (Optional) Register Jupyter kernel

```bash
python -m ipykernel install --user --name pinn --display-name "Python (pinn)"
```

### 4) Verify

```bash
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available())"
```

> Note: Do NOT install `torch/torchvision/torchaudio` via pip in this project.
> Mixing pip + conda for PyTorch often breaks `torch._C` (e.g., `NameError: _C is not defined`).

````

---

## environment.yml（只放 conda 要管的）
文件名：`environment.yml`

```yaml
name: pinn
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  # PyTorch (CUDA 11.8) - keep these in conda only
  - pytorch=2.1.2
  - torchvision=0.16.2
  - torchaudio=2.1.2
  - pytorch-cuda=11.8

  # nice-to-have (conda side)
  - pip
  - ipykernel
````

用法：

```bash
conda env create -f environment.yml
conda activate pinn
pip install -r requirements.txt
```

---

## requirements.txt（只放 pip 包）

文件名：`requirements.txt`

```txt
# pure-python / pip-side deps (DO NOT put torch/torchvision/torchaudio here)

numpy==1.26.4
pandas==2.2.2
matplotlib==3.9.4
openpyxl==3.1.5
h5py==3.14.0

# utils / notebooks
ipython==8.18.1
jupyter_client==8.6.3
jupyter_core==5.8.1
pyyaml==6.0.2
requests==2.32.5

# optional (if you actually use them)
nuitka==2.8.9
zstandard==0.25.0
```
