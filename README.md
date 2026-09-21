# MobileNetV2 Knowledge Distillation- PyTorch

## Overview
Knowledge distillation for efficient computer vision in the context of facial recognition and location — 
transferring the performance of a large teacher model into a smaller, faster student model better suited for operation on edge devices.

<p align="center"> <img src="src/Student_CNN/teacher_model/teacher_architecture.png" alt="Teacher Architecture" width="450"> <img src="src/Student_CNN/student_model/student_architecture.png" alt="Student Architecture" width="450"> 
</p> <p align="center"> <a href="#quick-start">Quick Start</a> • <a href="#training-results">Training Results</a> • <a href="#example-photos">Example Photos</a>

## Quick Start
### Requirements
For package requirements, see requirements.txt

### Download WIDER_FACE

Training Images:
- [Google Drive](https://drive.google.com/file/d/15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M/view?usp=sharing)
- [Hugging Face](https://huggingface.co/datasets/wider_face/blob/main/data/WIDER_train.zip)

Validation Images:
- [Google Drive](https://drive.google.com/file/d/1HIfDbVEWKmsYKJZm4lchTBDLW5N7dY5T/view?usp=sharing)
- [Hugging Face](https://huggingface.co/datasets/wider_face/blob/main/data/WIDER_val.zip)

Annotations:
- [WIDER_FACE](http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip)

> [!IMPORTANT]
> Use annotations.py to convert csv annotations to json-style format used by the dataset code.

### Open in visual studio code
Create a virtual environment, then run pip install -r requirements.txt.
Finally, run python -m pip install Student_CNN, then on the debugging tab you can select which code to run.
The default options are:
- Teacher Parameters
- Student Parameters
- Train Teacher
- Analyze Camera
- Analyze Image
- Train Student

## Training Results
> [!NOTE]
> Graphs start from 'epoch 0' instead of 'epoch 1', which is a bug.

### Training Statistics
| | Student without KD | Student with KD |
| --- | --- | --- |
| **Training Loss** <br> Model Value = 0.4977 | <p align="center"> <img src="model_no_kd/train/train_loss.jpg" alt="Training Loss no kd" width="300"> |
| **Validation Loss** <br> Model Value = 0.7907 |<p align="center"> <img src="model_no_kd/val/val_loss.jpg" alt="Validation Loss no kd" width="300"> |
| **F1** <br> Model Value = 0.5588 | <p align="center"> <img src="model_no_kd/val/f1.jpg" alt="F1 no kd" width="300"> |
| **Precision** <br> Model Value = 0.5636 | <p align="center"> <img src="model_no_kd/val/precision.jpg" alt="Precision no kd" width="300"> |
| **Recall** <br> Model Value = 0.5541 | <p align="center"> <img src="model_no_kd/val/recall.jpg" alt="Recall no kd" width="300"> |
| **Comments** | Training begins to become unstable at ~ epoch 16 (this is the saved model).<br> Precision oscillates as model begins to make more predictions, <br> which then get more precise. |

### Hyperparameters
| | Student without KD | Student with KD |
| --- | --- | --- |
| **Loss** | `lambda_box` = 2 <br> `lambda_noobj` = 2 <br> `lambda_wh` = 1 <br> |
| **Validation** | `confidence_threshold` = 0.8 <br> `nms_threshold` = 0.1 <br> `iou_threshold` = 0.3 |
| **Comments**| High confidence threshold gave balance between precision/recall for max f1. <br> More training information in log.txt |

## Example Photos
### Clear Face
| Clear Face | Output without KD | Output with KD |
| --- | --- | --- |
| <p align="center"> <img src="src/Student_CNN/test_images/soldier1.jpg" alt="Clear Face" width="100"> |<p align="center"> <img src="model_no_kd/test_images_output/output_soldier1.png" alt="Clear Face no_kd" width="300"> |
| **Comments** | Model can identify clear faces with a very large confidence and no noise|

### Obscured Face
| Obscured Face | Output without KD | Output with KD |
| --- | --- | --- |
| <p align="center"> <img src="src/Student_CNN/test_images/soldier2.jpg" alt="Obscured Face" width="200"> |<p align="center"> <img src="model_no_kd/test_images_output/output_soldier2.png" alt="Obscured Face no_kd" width="400"> |
| **Comments** | Model can still identify faces with a large confidence, but is less clear on the location of the bounding box|

### Grouped Faces
| Grouped Faces | Output without KD | Output with KD |
| --- | --- | --- |
| <p align="center"> <img src="src/Student_CNN/test_images/group.jpg" alt="Grouped Faces" width="200"> |<p align="center"> <img src="model_no_kd/test_images_output/output_group.png" alt="Grouped Faces no_kd" width="400"> |
| **Comments** | Difficult photo, and the model successfully identifies and localizes 18/22 faces.<br> There are a couple of high confidence false positives close to certain faces.|

### Animal Faces
| Grouped Faces | Output without KD | Output with KD |
| --- | --- | --- |
| <p align="center"> <img src="src/Student_CNN/test_images/handshake.jpg" alt="Animal Faces" width="200"> |<p align="center"> <img src="model_no_kd/test_images_output/output_handshake.png" alt="Animal Faces no_kd" width="400"> |
| **Comments** | Tested this photo as a joke, however it seems the model can identify dog faces too!|
