# YOLO Mask Detector

基于 YOLOv8 的轻量级口罩检测项目，支持两类目标：`masked`（佩戴口罩）和 `unmask`（未佩戴口罩）。

对应 GitHub 仓库：`https://github.com/hrh694958355-beep/yolo-mask-detector`

## 功能概览

- 统一命令行入口：训练、验证、图片推理、视频推理、GUI
- 单独脚本入口：`train.py`、`val.py`、`run_demo_image.py`、`run_demo_video.py`
- 图形化界面：`run_gui.py` + `gui_window.py`
- 默认预训练权重：`models/pretrained/yolov8n.pt`
- 可选已训练权重：`models/trained/best.pt`

## 项目结构

```text
yolo_mask_detector/
├─ app.py                   # 统一 CLI 入口
├─ train.py                 # 训练
├─ val.py                   # 验证
├─ run_demo_image.py        # 图片实时展示（不落盘）
├─ run_demo_video.py        # 视频实时展示（不落盘）
├─ run_gui.py               # GUI 启动入口
├─ gui_window.py            # GUI 主逻辑
├─ dataset.yaml             # 数据集配置（默认指向 mark-datas）
├─ requirements.txt
├─ assets/                  # 示例图片/视频
├─ mark-datas/              # 本地数据集（默认不入库）
├─ models/
│  ├─ pretrained/
│  └─ trained/
└─ outputs/                 # 训练与验证输出（默认不入库）
```

## 环境准备

```powershell
pip install -r requirements.txt
python smoke_test.py
python app.py --help
```

## 常用命令

### 1) 训练

```powershell
python app.py train --cache
```

默认训练参数：`imgsz=768`、`epochs=50`、`patience=30`、`close_mosaic=15`

### 2) 验证

```powershell
python app.py val
python val.py
```

### 3) 图片/视频演示（仅显示，不保存）

```powershell
python run_demo_image.py
python run_demo_video.py
```

- 图片演示：按任意键关闭窗口
- 视频演示：按 `q` 退出

### 4) GUI

```powershell
python run_gui.py
python app.py gui
```

## 数据与模型说明

- `mark-datas/` 与 `outputs/` 在 `.gitignore` 中默认忽略，避免提交大体积文件
- 如果需要共享训练权重，建议：
  - 小文件直接放到 `models/trained/`
  - 大文件使用 Git LFS 或云盘链接

## 新增脚本：评估、混淆矩阵与基准测试（scripts/）

本仓库新增了一组脚本放在 `scripts/` 下，用于自动化统计标签、运行验证、生成混淆矩阵、做推理基准并聚合输出为 JSON，方便复现实验与撰写报告。下面给出说明与 PowerShell 示例命令（在项目根运行）。

- `scripts/count_labels.py`
  - 功能：统计 `mark-datas/labels/train` 与 `mark-datas/labels/val` 下每类实例数。
  - 输出：`outputs/labels_count.json`
  - 示例：
    ```powershell
    python .\scripts\count_labels.py --labels-dir .\mark-datas\labels --out .\outputs\labels_count.json
    ```

- `scripts/val_with_yolo.py`
  - 功能：使用 `ultralytics`（或 `yolo` CLI）对指定权重做验证，保存 metrics 与 per-image predictions。
  - 输出：`outputs/val/{tag}_results.json`、`outputs/val/{tag}_preds.json`
  - 示例（预训练）：
    ```powershell
    python .\scripts\val_with_yolo.py --model .\models\pretrained\yolov8n.pt --data .\dataset.yaml --tag pretrained --out-dir .\outputs\val --imgsz 640 --batch 16
    ```
  - 示例（微调模型）：
    ```powershell
    python .\scripts\val_with_yolo.py --model .\models\trained\best.pt --data .\dataset.yaml --tag finetuned --out-dir .\outputs\val --imgsz 640 --batch 16
    ```

- `scripts/confusion_matrix.py`
  - 功能：基于 `*_preds.json` 与 YOLO txt 标注计算 IoU 配对，生成混淆矩阵 CSV 与摘要 JSON（TP/FP/FN）。
  - 输出：`outputs/val/{tag}_confusion.csv`、`outputs/val/{tag}_confusion.json`
  - 示例：
    ```powershell
    python .\scripts\confusion_matrix.py --preds .\outputs\val\finetuned_preds.json --labels .\mark-datas\labels\val --out .\outputs\val\finetuned_confusion --iou-thres 0.5
    ```

- `scripts/benchmark_inference.py`
  - 功能：对单张或一组图片重复推理，输出 avg/p50/p90 延迟与 FPS（支持 `cpu` 或 `cuda` 设备）。
  - 输出：`outputs/benchmark/{tag}_benchmark_*.json`
  - 示例（CPU）：
    ```powershell
    python .\scripts\benchmark_inference.py --model .\models\trained\best.pt --images .\assets\demo.jpg --runs 200 --warmup 10 --device cpu --out .\outputs\benchmark\finetuned_benchmark_cpu.json
    ```

- `scripts/collect_outputs.py`
  - 功能：自动收集 `outputs/` 下的 labels_count、val results、preds、confusion、benchmark 等文件，合并为 `outputs/collected_results.json` 便于粘贴与分析。
  - 示例：
    ```powershell
    python .\scripts\collect_outputs.py --out .\outputs\collected_results.json --outputs-dir .\outputs
    ```

注意事项：
- 请在项目根（含 `dataset.yaml` 与 `mark-datas`）下运行脚本。若 `val_with_yolo.py` 无法生成 preds，检查 `ultralytics` 是否安装或 `dataset.yaml` 中 `val` 字段路径是否正确且可访问。
- `outputs/` 默认在 `.gitignore` 中忽略，请在需要共享结果时仅上传 `scripts/` 与必要的结果摘要（JSON/CSV），不要提交大型模型权重或原始数据。

## 推送到 GitHub（建议步骤）

执行以下命令将新增脚本与 README 的更改提交并推送到远程仓库：

```powershell
# 1. 检查当前分支
git branch --show-current

# 2. 添加变更（仅添加脚本与 README，避免 outputs/）
git add scripts/README.md README.md
git add scripts/*.py

# 3. 提交
git commit -m "Add evaluation scripts: count_labels, val_with_yolo, confusion_matrix, benchmark_inference, collect_outputs; update README"

# 4. 推送到远程（假设分支为 main）
git push origin main
```

如果你的远程分支名是 `master` 或其他，请把 `main` 替换为实际分支名；首次推送可能会提示输入凭据或使用 Git credential manager/SSH Key。若需我帮你生成更细致的 commit message 或创建 PR 模板，也可以告诉我。

## 说明

- 英文/旧版说明可参考历史提交或自行维护 `README.zh-CN.md`
