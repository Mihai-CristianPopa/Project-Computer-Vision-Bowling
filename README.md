# Project-Computer-Vision-Bowling

### For installing the requirements
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

### Task 1

To run Task 1, there is a file in the task1 folder, task1.py
Just run that one file. Make sure that the PinDetection.pt is available too.

#### Output files of Task 1. Same for Tasks 2 and 3
Results for the train files will be available in:
results/train/Task1
Results for the fake-test files will be available in:
results/fake_test/Task1
Results for the test files will be available in:
Mihai_Popa_407/Task1 folder

####
Data used for fine-tuning the PinDetection.pt model available at:
[link text](https://universe.roboflow.com/lsc-kik8c/bowling-pin-detection)

### Task 2

To run Task 2, there is a file in the task2 folder, init.py
Just run that one file. Make sure that the BowlingBallDetection.pt is available too.

####
Data used for fine-tuning the PinDetection.pt model available at:
[link text](https://universe.roboflow.com/bsm-ecg3e/bowling-model/dataset/10)

### Task 3

To run Task 3, there is a file in the task3 folder, task3.py
Just run that one file. Make sure that the PinDetection.pt is available too.