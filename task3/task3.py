import cv2
import os

from ultralytics import YOLO
from typing import List, Dict, Literal

DEFAULT_CONFIDENCE = 0.31
MAX_DETECTIONS = 10
PIN_CLASS = [1]
CROP_FROM_EACH_SIDE = 300
CROP_FROM_BOTTOM = 140
SHOW_PLOT = False
PATH_TO_MODEL = "PinDetection.pt"
PATH_TO_TRAIN_VIDEOS = "data/train/Task3"
PATH_TO_TRAIN_PREDICTIONS = "results/train/Task3"
PATH_TO_TRAIN_GT = "data/train/Task3/ground-truth"
PATH_TO_FAKE_TEST_VIDEOS = "data/evaluation/fake_test/Task3"
PATH_TO_FAKE_TEST_PREDICTIONS = "results/fake_test/Task3"
PATH_TO_FAKE_TEST_GT = "data/evaluation/fake_test/ground-truth/Task3"
PATH_TO_TEST_VIDEOS = "data/test/Task3"
PATH_TO_TEST_GT = "data/test/ground-truth/Task3"
PATH_TO_TEST_PREDICTIONS = "Mihai_Popa_407/Task3"
DOT_TXT = ".txt"
DOT_MP4 = ".mp4" 
_PREDICTED = "_predicted"

RUNTYPE_TRAIN = "train"
RUNTYPE_FAKE_TEST = "fake_test"
RUNTYPE_TEST = "test"

Runtype = Literal["train", "fake_test", "test"]

def count_mp4_files(directory: str) -> int:
    """
    Count the number of .mp4 files in the specified directory.

    Parameters:
    directory (str): The path to the directory.

    Returns:
    int: The number of .mp4 files in the directory.
    """
    return len([file for file in os.listdir(directory) if file.endswith('.mp4')])

def is_file_of_type(filename: str, ending: str) -> bool:
    return filename.endswith(ending)

def load_results(directory: str) -> list:
    queries = []
    for filename in os.listdir(directory):
        if is_file_of_type(filename, DOT_TXT):
            with open(os.path.join(directory, filename), 'r') as file:
                query = [line.strip() for line in file]
            queries.append(" | ".join(query))
    return queries

def calculate_accuracy(predicted_truths: list, ground_truths: list) -> float:
    sum = 0
    index = 1
    for prediction, truth in zip(predicted_truths, ground_truths):
        if prediction == truth:
            sum += 1
        else:
            print(index)
        index += 1        
    return sum / len(predicted_truths)

def crop_frame(frame: cv2.typing.MatLike, crop_sides: int, crop_bottom: int) -> cv2.typing.MatLike:
    height, width, _ = frame.shape
    new_width = width - 2 * crop_sides
    new_height = height - crop_bottom
    cropped_frame = frame[:new_height, crop_sides:crop_sides + new_width]
    cropped_frame = cv2.resize(cropped_frame, (640, 640))
    return cropped_frame

def get_median(list: list) -> float:
    list.sort()
    length = len(list)
    if length % 2 == 0:
        return (list[length//2] + list[length//2 - 1]) / 2
    return list[length//2]


def get_number_of_standing_pins(model: YOLO, image: cv2.typing.MatLike, classes=PIN_CLASS, conf=DEFAULT_CONFIDENCE, max_det=MAX_DETECTIONS, show_plot=SHOW_PLOT):
    results = model.predict(image, classes=classes, conf=conf, max_det=max_det)
    # Display the annotated frame
    if show_plot:
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return len(results[0].boxes.xyxy.cpu().numpy())


def task3(runtype: Runtype=RUNTYPE_TRAIN, show_plot: bool=SHOW_PLOT):
    if runtype == RUNTYPE_TRAIN:
        PATH_TO_VIDEOS = PATH_TO_TRAIN_VIDEOS
        PATH_TO_PREDICTIONS = PATH_TO_TRAIN_PREDICTIONS
        PATH_TO_GT = PATH_TO_TRAIN_GT
        EVAL = True
    elif runtype == RUNTYPE_FAKE_TEST:
        PATH_TO_VIDEOS = PATH_TO_FAKE_TEST_VIDEOS
        PATH_TO_PREDICTIONS = PATH_TO_FAKE_TEST_PREDICTIONS
        PATH_TO_GT = PATH_TO_FAKE_TEST_GT
        EVAL = True
    elif runtype == RUNTYPE_TEST:
        PATH_TO_VIDEOS = PATH_TO_TEST_VIDEOS
        PATH_TO_PREDICTIONS = PATH_TO_TEST_PREDICTIONS
        PATH_TO_GT = PATH_TO_TEST_GT
        EVAL = False
    yolo_model = YOLO(PATH_TO_MODEL)
    for i in range (1, count_mp4_files(PATH_TO_VIDEOS) + 1):
        video_path = f"{PATH_TO_VIDEOS}/0{i}{DOT_MP4}" if i < 10 else f"{PATH_TO_VIDEOS}/{i}{DOT_MP4}"
        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            print("Could not open video")
            return 1
        # Read first frame.
        ok, first_frame = video.read()
        if not ok:
            return 1

        first_frame = crop_frame(first_frame, CROP_FROM_EACH_SIDE, CROP_FROM_BOTTOM)
        
        num_pins_first_frame = get_number_of_standing_pins(yolo_model, first_frame, classes=PIN_CLASS, conf=DEFAULT_CONFIDENCE, max_det=MAX_DETECTIONS, show_plot=show_plot)
        frame_list = []
        while True:
            # Read a new frame
            ok, frame = video.read()
            if ok:
                frame_list.append(frame)
            else:    
                break
        video.release()    
        num_pins_last_frame = get_number_of_standing_pins(yolo_model, frame_list[-1], classes=PIN_CLASS, conf=DEFAULT_CONFIDENCE, max_det=MAX_DETECTIONS, show_plot=show_plot)
        query_name = f"0{i}{_PREDICTED}{DOT_TXT}" if i <10 else f"{i}{_PREDICTED}{DOT_TXT}"
        
        file_path = os.path.join(os.getcwd(), PATH_TO_PREDICTIONS, query_name)

        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))  
                    
        with open(file_path, 'w') as file:
            file.write(f"{num_pins_first_frame}\n")
            file.write(f"{num_pins_last_frame}")
    
    if EVAL:        
        predicted_truths = load_results(PATH_TO_PREDICTIONS)
        ground_truths = load_results(PATH_TO_GT)
        acc = calculate_accuracy(predicted_truths, ground_truths)
        print(acc)
    return 0

def main():
    # return_code = task3(RUNTYPE_TRAIN, False)
    # return_code = task3(RUNTYPE_FAKE_TEST, False)
    return_code = task3(RUNTYPE_TEST, True)
    if return_code == 0:
        print("Task3 completed successfully")
    elif return_code == 1:
        print("Error in loading the video")

if __name__ == "__main__":
    main()     
    