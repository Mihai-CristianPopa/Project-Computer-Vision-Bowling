# Project-Computer-Vision-Bowling

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [How to Run](#how-to-run)
  - [Task 1](#task-1)
  - [Task 2](#task-2)
  - [Task 3](#task-3)
- [Data Storage](#data-storage)
- [Datasets](#datasets)
- [Results and Evaluation](#results-and-evaluation)

## Description
The project contains three tasks:

1. **Task 1**:
   - You receive four images of different bowling lanes with the full configuration of pins.
   - You receive an image with the pins after the bowling ball has hit them.
   - You receive a txt file, called query, which contains, on the first line, the number of pins that we want to predict on and on the next line, indices of the pins for which we want to predict the positions.
   - The task is to respond to the query, for each of the pin indices with 0 if the pin has fallen or 1 if the pin is standing in the image.

2. **Task 2**:
   - You receive a video of a bowling lane with the ball rolling towards the pins.
   - You receive a text file containing the number of frames of the video and the initial bounding box of the ball to be tracked
   - The task is to track the ball and predict its position in each frame of the video.
   - You need to output the bounding box coordinates of the ball in each frame.

3. **Task 3**:
   - You receive a video of a bowling lane with the ball rolling towards the pins.
   - The task is to detect the pins and predict how many pins are standing in the begining of the video and how many pins are standing in the end of the video, after the ball has hit them.

## Installation

### Installing the Requirements
Automatically:
```sh
pip install -r requirements.txt
```

Listed requirements:
```sh
certifi==2024.8.30
charset-normalizer==3.3.2
colorama==0.4.6
contourpy==1.3.0
cycler==0.12.1
filelock==3.15.4
filetype==1.2.0
fonttools==4.53.1
fsspec==2024.6.1
idna==3.7
intel-openmp==2021.4.0
Jinja2==3.1.4
kiwisolver==1.4.7
lapx==0.5.10.post1
MarkupSafe==2.1.5
matplotlib==3.9.2
mkl==2021.4.0
mpmath==1.3.0
networkx==3.3
numpy==1.26.4
opencv-contrib-python==4.10.0.84
opencv-python==4.10.0.84
packaging==24.1
pandas==2.2.2
pillow==10.4.0
psutil==6.0.0
py-cpuinfo==9.0.0
pyparsing==3.1.4
python-dateutil==2.9.0.post0
python-dotenv==1.0.1
pytz==2024.1
PyYAML==6.0.2
requests==2.32.3
requests-toolbelt==1.0.0
roboflow==1.1.44
scipy==1.14.1
seaborn==0.13.2
six==1.16.0
sympy==1.13.2
tbb==2021.13.1
torch==2.3.1
torchvision==0.18.1
tqdm==4.66.5
typing_extensions==4.12.2
tzdata==2024.1
ultralytics==8.2.87
ultralytics-thop==2.0.6
urllib3==2.2.2
```

## How to run
### Task 1

To run Task 1, there is a file in the task1 folder, task1.py.
Just run that one file. Make sure that the PinDetection.pt is available too.

#### Options for Running Task 1

The `task1` function can be run with different options based on the `runtype` and `display_mode` parameters:

- **Runtype Options**:
  - `train`: Runs the task using the training dataset.
  - `fake_test`: Runs the task using the fake test dataset.
  - `test`: Runs the task using the test dataset.

- **Display Mode Options**:
  - `save`: Saves the images with bounding boxes to the specified directory.
  - `display`: Displays the images with bounding boxes.

#### Example Usage

```python
# Run Task 1 with training data and save the results
task1(runtype="train", display_mode="save")

# Run Task 1 with fake test data and display the results
task1(runtype="fake_test", display_mode="display")

# Run Task 1 with test data and save the results
task1(runtype="test", display_mode="save")
```

### Task 2

To run Task 2, there is a file in the task2 folder, init.py.
Just run that one file. Make sure that the BowlingBallDetection.pt is available too.

#### Options for Running Task 2

The `task2` function can be run with different options based on the `runtype` and `show_plot` parameters:

- **Runtype Options**:
  - `train`: Runs the task using the training dataset.
  - `fake_test`: Runs the task using the fake test dataset.
  - `test`: Runs the task using the test dataset.

- **Show Plot Option**:
  - `True`: Displays the frames with bounding boxes during processing.
  - `False`: Does not display the frames during processing.

#### Example Usage

```python
# Run Task 2 with training data and do not display the frames
task2(runtype="train", show_plot=False)

# Run Task 2 with fake test data and display the frames
task2(runtype="fake_test", show_plot=True)

# Run Task 2 with test data and display the frames
task2(runtype="test", show_plot=True)
```
### Task 3

To run Task 3, there is a file in the task3 folder, task3.py.
Just run that one file. Make sure that the PinDetection.pt is available too.

#### Options for Running Task 3

The `task3` function can be run with different options based on the `runtype` and `show_plot` parameters:

- **Runtype Options**:
  - `train`: Runs the task using the training dataset.
  - `fake_test`: Runs the task using the fake test dataset.
  - `test`: Runs the task using the test dataset.

- **Show Plot Option**:
  - `True`: Displays the frames with bounding boxes during processing.
  - `False`: Does not display the frames during processing.

#### Example Usage

```python
# Run Task 3 with training data and do not display the frames
task3(runtype="train", show_plot=False)

# Run Task 3 with fake test data and display the frames
task3(runtype="fake_test", show_plot=True)

# Run Task 3 with test data and display the frames
task3(runtype="test", show_plot=True)
```

## Data Storage

### To be processed data
All data used in the tasks is available in the data folder.
There data is put in three categories: train, evaluation and test.
For train and evaluation ground-truth is also available so for those two categories
numerical evaluation is available too in the tasks.

### Task 1
- **Train Results**: `results/train/Task1`
- **Fake-Test Results**: `results/fake_test/Task1`
- **Test Results**: `Mihai_Popa_407/Task1`
- **Images with Pin Bounding Boxes**: `save_img/Task1`
The images with pin bounding boxes are put initially in three categories based on the processed data (train/evaluation/test). And then for each of these there are the following folders:
- **Lane Images with masked pin bounding boxes**: `save_img/Task1/<category>/lane_images/obstructed`
- **Lane Images with unmasked pin bounding boxes**: `save_img/Task1/<category>/lane_images/unobstructed`
- **Lane Images with not well identified bottom**: `save_img/Task1/<category>/lane_images/unbalanced`
- **Predicted Images pin bounding boxes**: `save_images/Task1/<category>/predicted_images`s

### Task 2
- **Train Results**: `results/train/Task2`
- **Fake-Test Results**: `results/fake_test/Task2`
- **Test Results**: `Mihai_Popa_407/Task2`

### Task 3
- **Train Results**: `results/train/Task3`
- **Fake-Test Results**: `results/fake_test/Task3`
- **Test Results**: `Mihai_Popa_407/Task3`

## Datasets

### Task 1
Data used for fine-tuning the PinDetection.pt model available at:
[Pin Detection Dataset](https://universe.roboflow.com/lsc-kik8c/bowling-pin-detection)

### Task 2
Data used for fine-tuning the BowlingBallDetection.pt model available at:
[Bowling Ball Detection Dataset](https://universe.roboflow.com/bsm-ecg3e/bowling-model/dataset/10)

### Task 3
Data used for fine-tuning the PinDetection.pt model available at:
[Pin Detection Dataset](https://universe.roboflow.com/lsc-kik8c/bowling-pin-detection)

## Results and Evaluation

### Task 1
Each predicted query is compared to the associated ground truth query for the image. If all pins' positions have been predicted correctly then the predicted query is considered correct, otherwise the predicted query is considered incorrect. The final result is the number of correctly predicted queries over the total number of queries.

### Task 2
The result of the tracking is a file that contains the coordinates of the bounding box for each frame of the video where the bowling ball is visible. For a video, the bowling ball is considered correctly tracked if it has been correctly tracked in more than 80% of the frames of the video. Correctly tracked for a frame, means that the IoU(Intersection over Union) between the predicted bounding box and the ground truth bounding box of the tracked object is larger than 30%.

### Task 3
Each of the text files is compared with the associated ground truth text file. If both the initial number of predicted standing pins and the final number of predicted standing pins are equal to the ones from the ground truth file, then the predicted text file is considered correct. In the end the result is the number of correctly predicted videos over the total number of videos.