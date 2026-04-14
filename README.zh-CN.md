# YOLO 口罩识别器

这是一个轻量级 YOLOv8 项目，专注于口罩识别（`masked` / `unmask`）。

## 项目结构

- `app.py`：统一 CLI，支持 train/val/predict/video/gui
- `train.py`：独立训练入口
- `val.py`：独立验证入口
- `run_demo_image.py`：对 `assets/demo.jpg` 进行实时可视化检测（不落盘）
- `run_demo_video.py`：对 `assets/demo.mp4` 进行实时可视化检测（不落盘）
- `run_gui.py`、`gui_window.py`：简化图形界面（图片与视频检测）
- `dataset.yaml`：数据集配置（相对路径：`mark-datas`）
- `models/pretrained/yolov8n.pt`：基础模型
- `models/trained/best.pt`：兜底训练权重
- `outputs/`：训练/验证输出目录

## 快速开始

```powershell
cd "your path\yolo_mask_detector"
pip install -r requirements.txt
python smoke_test.py
python app.py --help
```

## 训练

```powershell
python app.py train --cache
```

默认训练参数：

- `imgsz=768`
- `epochs=50`
- `patience=30`
- `close_mosaic=15`

## 验证

```powershell
python app.py val
python val.py
```

## 实时演示（不生成输出文件）

```powershell
python run_demo_image.py
python run_demo_video.py
```

- `run_demo_image.py`：弹出单图检测窗口，按任意键关闭。
- `run_demo_video.py`：弹出视频检测窗口，按 `q` 退出。

## 图形界面

```powershell
python run_gui.py
python app.py gui
```

## English Version

请查看 `README.md`。

