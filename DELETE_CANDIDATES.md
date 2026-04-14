# Delete Candidates (Do Not Delete Automatically)

This list is generated after migrating minimal runnable assets into `mini_yolo_app`.

## Migrated replacements

- `yolov8-42/yolov8n.pt` -> `mini_yolo_app/models/pretrained/yolov8n.pt`
- `yolov8-42/42_demo/runs/detect/train5/weights/best.pt` -> `mini_yolo_app/models/trained/best.pt`
- `yolov8-42/42_demo/images/resources/demo.jpg` -> `mini_yolo_app/assets/demo.jpg`
- `yolov8-42/42_demo/images/resources/demo.mp4` -> `mini_yolo_app/assets/demo.mp4`

## Candidate files/directories to remove later

1. `42_demo/start_train.py` (replaced by `mini_yolo_app/app.py train`)
2. `42_demo/start_val.py` (replaced by `mini_yolo_app/app.py val`)
3. `42_demo/start_single_detect.py` (replaced by `mini_yolo_app/app.py predict`)
4. `42_demo/start_webcam.py` (replaced by `mini_yolo_app/app.py video`)
5. `42_demo/start_window.py` (optional GUI, not needed for minimal run)
6. `42_demo/images/tmp/` (temporary artifacts)
7. `42_demo/record/` (record outputs)
8. `42_demo/runs/` (legacy run outputs; keep if you still need old experiments)
9. `runs/` (root-level legacy run outputs; keep if still referenced)
10. `ultralytics/` (local source tree, only removable if you switch fully to pip package and stop local code edits)
11. `docs/`, `examples/`, `docker/`, `tests/` (non-essential for minimal app runtime)

## Safety checklist before deleting

- Confirm training still works from `mini_yolo_app/app.py train`
- Confirm validation works from `mini_yolo_app/app.py val`
- Confirm image/video inference works from `mini_yolo_app/app.py predict` and `mini_yolo_app/app.py video`
- Confirm no external script imports paths under `42_demo/` or `ultralytics/`

