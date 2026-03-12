# YOLOFuse-events

This repository contains the official code for our paper:

**Understanding Fundamental Modality Advantage in RGB and Event-based Hybrid Data**

This repository contains our multimodal object detection codebase built on top of [YOLOFuse](https://github.com/WangQvQ/YOLOFuse), which itself is based on the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) framework. Our implementation extends YOLOFuse for aligned RGB-event object detection and related experiments.

## Installation

Clone the repository and install it in editable mode:

```bash
git clone git@github.com:wshs7/YOLOFuse-events.git
cd YOLOFuse-events
pip install -e .
```

It is recommended to use a clean Python environment with the dependencies required by Ultralytics/YOLOFuse.

---

## Dataset Download

Please download the dataset from Google Drive:

**MAD-Drone:** https://drive.google.com/file/d/1Rw9r8QFUAiblSgn2JlC5Ajm_cFjYpz1b/view?usp=drive_link

**MAD-Drone-real:** https://drive.google.com/file/d/1r5QeH03Kx2P8Dq-i1DA9986BDFogycLz/view?usp=drive_link

After downloading, extract the dataset to your preferred location.

---

## Dataset Structure

The dataset should follow the structure below:

```text
dataset_root/
├── images/
│   ├── train/
│   ├── val/
│   └── test/              # optional
├── imagesIR/              # optional event modality
│   ├── train/
│   ├── val/
│   └── test/              # optional
└── labels/
    ├── train/
    ├── val/
    └── test/              # optional
```


---

## Configure `data_yaml/`

Before running training or validation, please edit the YAML files in the `data_yaml/` directory and set the correct dataset paths.

A typical directory layout is:

```text
data_yaml/
├── frame.yaml
├── event.yaml
└── multi.yaml
```

A typical YAML file looks like this:

```yaml
path: /path/to/dataset_root
train: images/train
val: images/val
test: images/test    # optional

names:
  0: object
```

Please modify the paths according to your local machine. Make sure that:

- the dataset root is correct
- the label files follow the standard YOLO directory structure

---

## Training

After preparing the dataset and editing the YAML file, you can start training multi-modality model with:


```bash
python train_multi.py --data data_yaml/multi.yaml
```

For the frame-only and event-only modaity, you can train with:

```bash
python train_frame.py --data data_yaml/frame.yaml
python train_event.py --data data_yaml/event.yaml
```


---

## Validation

To run validation, use:

```bash
python val_dual.py --data data_yaml/multi.yaml
```


## Citation

If you use this repository in your research, please cite our work and also acknowledge YOLOFuse.

### Base framework

```bibtex
@misc{yolofuse,
  title={YOLOFuse},
  author={WangQvQ},
  howpublished={\url{https://github.com/WangQvQ/YOLOFuse}}
}
```

You may also cite the Ultralytics YOLO framework if appropriate.

---

