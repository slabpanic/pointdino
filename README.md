# PointDINO

本仓库为 PointDINO 关键点检测工程，基于本地 MMDetection runtime，使用 DINO
query 直接预测单类目标的二维点坐标和置信度。

## 快速开始

依赖安装（Python 3.12；MMCV 使用 CUDA 12.1 / PyTorch 2.4.1 的官方 wheel）：

```bash
uv python install 3.12
uv venv --python 3.12 --clear
uv sync --no-dev
```

默认依赖只包含推理运行时，安装包不包含项目 YAML 和训练入口。将仓库作为 uv 依赖导入时：

```bash
uv add git+<仓库地址>
```

`import pointdino` 和 `from pointdino import PointDINOPredictor` 都采用按需导入，
不会在导入阶段加载训练、评估、可视化和数据集扩展模块。训练必须在 clone 的源码仓库中
进行，并安装训练扩展：

```bash
uv sync --extra train
```

验证环境：

```bash
uv run --no-dev python -c "import torch, mmcv, mmengine; print(torch.__version__, torch.version.cuda, mmcv.__version__, mmengine.__version__)"
```

## 配置

源码仓库中的运行参数写在 `config/` 中，不使用环境变量，也不通过命令行覆盖参数。
配置文件只有三个；这些文件不属于安装包：

```text
config/
├── common.yaml       公共模型、运行时和测试数据配置
├── train.yaml        训练数据、优化器和训练输出配置
└── inference.yaml    推理输入、权重、阈值和推理输出配置
```

数据目录按下面的结构放置，并在 [config/common.yaml](config/common.yaml) 中修改
`data_root`：

```text
data/
├── train_point.json
├── val_point.json
└── export/
    └── images...
```

训练前在 [config/train.yaml](config/train.yaml) 中设置 `load_from`（没有初始化权重
时保持 `null`）。训练日志、检查点和 MMEngine 运行数据统一写入 `output/train/`。

推理前在 [config/inference.yaml](config/inference.yaml) 中设置：

```yaml
inference:
  checkpoint: ./output/train/best_point_f1@10px_epoch_75.pth
  input: ./data/inference
  score_threshold: 0.288
  output: ./output/inference/prediction.json
  visualization_dir: ./output/inference/vis
```

## 项目结构

```text
src/pointdino/
├── inference.py        可复用推理 API、图片枚举和结果结构
├── models/             PointDINO 注册别名
└── evaluation.py       点检测评估器注册

tools/
├── config_loader.py     仓库专用 YAML 合并和配置加载
├── train.py             仓库专用训练入口
└── predict.py           仓库专用批量推理入口

config/
├── common.yaml         公共模型和运行时配置
├── train.yaml          训练配置
└── inference.yaml      推理配置

examples/
└── inference.py        读取 YAML 并调用推理 API 的示例
```

安装包只提供参数驱动的推理 API，不提供 YAML 读取接口。仓库内的 `tools/` 和
`examples/` 脚本才负责读取 YAML；修改仓库运行行为时直接编辑对应配置文件。

## 数据格式

数据集使用 MMDetection `BaseDetDataset` 格式。每个标注 JSON 的 `images` 项需要提供
`file_name`、`height`、`width`，每个 `annotations` 项至少提供：

```json
{
  "point": [123.4, 567.8],
  "point_label": 0,
  "ignore_flag": 0
}
```

`config/common.yaml` 中的 `data_root` 应指向包含标注和图片目录的父目录：

```text
data_root/
├── train_point.json
├── val_point.json
└── export/
    └── images...
```

如果图片不在 `export/` 子目录，可在训练和公共配置的数据集项中把
`data_prefix.img_path` 改为空字符串。

## 训练

```bash
uv run python -m tools.train
```

混合精度、断点续训、学习率自动缩放等选项均在 `config/train.yaml` 中配置。训练输出
包括日志、检查点、配置副本和可视化运行数据，统一位于 `output/train/`。

## 推理

把待推理图片放入 `config/inference.yaml` 中 `inference.input` 指定的文件或目录，
然后运行：

```bash
uv run --no-dev python -m tools.predict
```

预测 JSON 和可视化图片分别写入 `inference.output` 与
`inference.visualization_dir`，均位于 `output/` 下。推理会复用公共配置中的 test
pipeline，并将点坐标还原到输入图片原始像素坐标系。所有推理输出都位于
`output/inference/`。

## Python API

推理 API 不读取 YAML、环境变量或命令行参数，也不依赖包内配置文件。调用方需要准备
模型配置对象（MMEngine `Config` 或普通 mapping），再把它和其他运行参数显式传入：

```python
import cv2
from pointdino import PointDINOPredictor

predictor = PointDINOPredictor(
    model_config,                         # caller-provided Config or dict
    checkpoint_path="/path/to/model.pth",
    score_threshold=0.288,
    device="cuda:0",
    log_level="INFO",
    log_file=None,
)
result = predictor.predict(cv2.imread("data/inference/image.png"))
```

完整的 YAML 读取和单图推理演示见
[`examples/inference.py`](examples/inference.py)：

```bash
uv run --no-dev python -m examples.inference
```

返回：

```python
{"points": [[x, y], ...], "scores": [score, ...]}
```

训练和推理入口的日志统一使用 Loguru，日志级别和日志文件路径由 YAML 配置。
三个 YAML 只存在于源码 checkout 的根目录 `config/`。scipy、shapely、pycocotools、
matplotlib、tqdm 等训练侧依赖由 `train` extra 显式声明；框架本身的传递依赖仍由
MMCV/MMEngine 管理。
