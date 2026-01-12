# 数字孪生AI项目

## 项目简介

本项目是一个数字孪生AI解决方案，旨在通过数据驱动的算法模型对系统进行智能分析和预测，并通过直观的可视化界面展示运行状态和预测结果。项目分为核心算法部分和前端可视化部分，实现了从数据处理、模型训练到结果展示的全流程。

## 项目结构

```
project/
├── algorithm/            # 核心算法部分，包含数据驱动模型
│   ├── 单制冷/           # 单制冷模式下的AI算法和数据
│   ├── 单制热/           # 单制热模式下的AI算法和数据
│   └── 混合制热/         # 混合制热模式下的AI算法和数据
├── visualization/        # 前端可视化界面，用于展示数字孪生数据
│   ├── front/            # 前端代码 (HTML, CSS, JS)
│   ├── sample_data.csv   # 示例数据文件
│   ├── sample_list_data.csv # 示例列表数据文件
│   └── server.exe        # Windows平台下的前端服务可执行文件
└── .gitignore            # Git版本控制忽略文件
└── README.md             # 项目说明文档
```

## 算法部分 (`algorithm/`)

算法部分主要包含针对不同制冷/制热模式的数据驱动模型。每个子目录都包含了相应的代码、数据和训练好的模型。

### 单制冷 (`algorithm/单制冷/`)

- **`tiny_model/`**: 包含模型定义和训练结果。
- **`train_case/`**: 训练数据，格式为 `.h5`。
- **`test_case/`**: 测试数据，格式为 `.h5`。
- **`utils/`**: 包含模型 (`model.py`, `network.py`) 和数据生成 (`generate_dataset.ipynb`)、模型训练 (`train_model.py`) 相关的Python脚本。

**技术栈**: Python, PyTorch , Jupyter Notebook, HDF5 

### 单制热 (`algorithm/单制热/`)

- **`code/`**: 包含 Jupyter Notebook (`ptc_heating_code.ipynb`)，用于模型开发和分析。
- **`data/`**: 训练和测试数据，格式为 `.npy` 和 `.xlsx`。
- **`model/`**: 训练好的模型 (`best_model.pt`)。

**技术栈**: Python, PyTorch , Jupyter Notebook, NumPy, Pandas

### 混合制热 (`algorithm/混合制热/`)

- **`code/`**: 包含 Jupyter Notebook (`hybrid_heating_pinn_v2.ipynb`)，实现了基于物理信息神经网络 (PINN) 的混合制热模型。
- **`data/`**: 训练和测试数据，格式为 `.npy` 和 `.xlsx`。
- **`model/`**: 训练好的模型 (`best_balanced_hybrid_pinn_v2.pt`)。

**技术栈**: Python, PyTorch (推测), Jupyter Notebook, NumPy, Pandas, PINN

## 可视化部分 (`visualization/`)

可视化部分提供了一个前端界面，用于展示数字孪生系统的实时数据和算法模型的输出。它包含一个基于 HTML/CSS/JavaScript 的前端应用和一个 Windows 可执行文件作为服务。

- **`front/`**: 包含前端页面的所有静态资源，如 `index.html`, `js/`, `lib/`, `storage/` 等。
- **`sample_data.csv`**, **`sample_list_data.csv`**: 示例数据文件，用于前端展示。
- **`server.exe`**: 一个 Windows 可执行文件，推测是用于启动本地服务器或提供数据接口。

**技术栈**: HTML, CSS, JavaScript, CSV, Windows Executable

## 环境配置与运行

### 算法部分环境配置

建议使用 `conda` 或 `pip` 创建独立的 Python 虚拟环境。

1. **创建虚拟环境** (以 `conda` 为例):
   ```bash
   conda create -n digital_twin_ai python=3.9  # 或其他版本
   conda activate digital_twin_ai
   ```

2. **安装依赖**:
   由于没有提供 `requirements.txt`，需要根据 Jupyter Notebook 和 Python 脚本中的导入语句手动安装。
   常见的依赖可能包括：
   ```bash
   pip install torch torchvision torchaudio  # PyTorch及其相关库
   pip install numpy pandas h5py openpyxl   # 数据处理库
   pip install jupyterlab                   # 如果需要运行Jupyter Notebook
   ```
   **注意**: 请检查各个算法子目录中的 `.ipynb` 文件和 `.py` 文件，以获取准确的依赖列表。

### 可视化部分运行

1. **启动前端服务**:
   - 如果在 Windows 环境下，可以直接运行 `visualization/server.exe` 来启动本地服务。
   - 如果在其他操作系统或需要更灵活的部署，可能需要分析 `server.exe` 的功能，或者手动配置一个静态文件服务器（如 `nginx`, `apache`, `http-server` npm包）来托管 `visualization/front` 目录。

2. **访问界面**:
   服务启动后，通过浏览器访问相应的地址（通常是 `http://localhost:port/index.html`）。

## 使用说明

1. **数据准备**：将您的实际数据按照 `algorithm/` 目录下各模块的数据格式进行准备。
2. **模型训练**：根据需要，运行 `algorithm/` 目录下相应的 Jupyter Notebook 或 Python 脚本进行模型训练或微调。
3. **数据接口**：确保算法模型的输出能够被可视化部分正确读取（可能需要修改 `server.exe` 或前端代码的数据加载逻辑）。
4. **界面展示**：启动可视化服务，在浏览器中查看数字孪生系统的运行状态和预测结果。

## 贡献

欢迎对本项目进行贡献。请遵循以下步骤：
1. Fork 本仓库。
2. 创建新的功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request。

## 许可证

本项目采用 MIT 许可证。详情请参阅 `LICENSE` 文件。

---

**作者**: Peng
**日期**: 2026年1月12日
