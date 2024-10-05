import cv2

from ultralytics import YOLO
from ultralytics.engine.results import Results
import os
from typing import List, Dict, Tuple, Literal

PATH_TO_MODEL = "BowlingBallDetection.pt"
PATH_TO_TRAIN_DATA = "data/train/Task2"
PATH_TO_TRAIN_PREDICTIONS = "results/train/Task2"
PATH_TO_TRAIN_GT = "data/train/Task2/ground-truth"
PATH_TO_FAKE_TEST_DATA = "data/evaluation/fake_test/Task2"
PATH_TO_FAKE_TEST_PREDICTIONS = "results/fake_test/Task2"
PATH_TO_FAKE_TEST_GT = "data/evaluation/fake_test/ground-truth/Task2"
PATH_TO_TEST_DATA = "data/test/Task2"
PATH_TO_TEST_GT = "data/test/ground-truth/Task2"
PATH_TO_TEST_PREDICTIONS = "Mihai_Popa_407/Task2"
DOT_TXT = ".txt"
DOT_MP4 = ".mp4"
_GT = "_gt"
_PREDICTED = "_predicted"
SHOW_PLOT = False
BALL_CLASS = [0]
DEFAULT_CONFIDENCE = 0.1
MAX_DET = 1

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

def create_tuple_bbox_from_results_v2(results: Results, average_bb: tuple, num_of_mock_bb: int, frame: cv2.typing.MatLike, show_plot=SHOW_PLOT) ->Tuple[tuple, tuple]:
    if results[0]:
        num_of_mock_bb = 0
        curr_bbox = tuple(results[0].boxes.xyxy.cpu().numpy().tolist()[0])
        average_bb = tuple((avg_el + curr_el) / 2 for avg_el, curr_el in zip(average_bb, curr_bbox))
        return curr_bbox, average_bb, num_of_mock_bb
    else:
        new_bbox = average_bb
        num_of_mock_bb += 1
        if show_plot:
            cv2.rectangle(frame, (int(new_bbox[0]), int(new_bbox[1])), (int(new_bbox[2]), int(new_bbox[3])), (0, 255, 0), 2)
            cv2.waitKey(25)
        return new_bbox, average_bb, num_of_mock_bb

# Function to read bounding boxes from a text file
def read_bounding_boxes(file_path):
    bounding_boxes = []
    with open(file_path, 'r') as file:
        first_line = file.readline().strip().split()
        num_of_frames = int(first_line[0])
        for line in file:
            parts = line.strip().split()
            if len(parts) == 5:
                frame_index = int(parts[0])
                x = int(parts[1])
                y = int(parts[2])
                xmax = int(parts[3])
                ymax = int(parts[4])
                bounding_boxes.append((x, y, xmax, ymax))
    return num_of_frames, bounding_boxes

def compute_iou(bb1: tuple, bb2: tuple) -> float:
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    bb1 (tuple): A tuple containing (left, top, right, bottom) coordinates of the first bounding box.
    bb2 (tuple): A tuple containing (left, top, right, bottom) coordinates of the second bounding box.

    Returns:
    float: The IoU of the two bounding boxes.
    """
    if  bb1[0] >= bb1[2] or bb1[1] >= bb1[3] or bb2[0] >= bb2[2] or bb2[1] >= bb2[3]:
        return 0.0


    # Determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    # Check if there is an intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def task2(runtype: Runtype = RUNTYPE_TRAIN, show_plot=SHOW_PLOT):
    if runtype == RUNTYPE_TRAIN:
        PATH_TO_DATA = PATH_TO_TRAIN_DATA
        PATH_TO_PREDICTIONS = PATH_TO_TRAIN_PREDICTIONS
        PATH_TO_GT = PATH_TO_TRAIN_GT
        EVAL = True
    elif runtype == RUNTYPE_FAKE_TEST:
        PATH_TO_DATA = PATH_TO_FAKE_TEST_DATA
        PATH_TO_PREDICTIONS = PATH_TO_FAKE_TEST_PREDICTIONS
        PATH_TO_GT = PATH_TO_FAKE_TEST_GT
        EVAL = True
    elif runtype == RUNTYPE_TEST:
        PATH_TO_DATA = PATH_TO_TEST_DATA
        PATH_TO_PREDICTIONS = PATH_TO_TEST_PREDICTIONS
        PATH_TO_GT = PATH_TO_TEST_GT
        EVAL = False
    # # Load the YOLOv8 model
    model = YOLO(PATH_TO_MODEL)
    dict_of_results = {}
    for i in range (1, count_mp4_files(PATH_TO_DATA) + 1):
        # Open the video file
        video_id = f"0{i}" if i < 10 else f"{i}"
        video_path = f"{PATH_TO_DATA}/{video_id}{DOT_MP4}"
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Could not open video")
            return 1
        is_first_frame = True
        # Loop through the video frames
        prediction_bb = []
        num_of_mock_bb = 0
        frame_count = 0
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()
            if success:
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                if is_first_frame: # Initialize with the input bounding box
                    file_path = f"{PATH_TO_DATA}/{video_id}{DOT_TXT}"
                    N, first_frame_bb_list = read_bounding_boxes(file_path)
                    first_frame_bb = first_frame_bb_list[0]
                    average_bb = first_frame_bb
                    prediction_bb.append(first_frame_bb)
                    if show_plot:
                        cv2.rectangle(frame, (first_frame_bb[0], first_frame_bb[1]), (first_frame_bb[2], first_frame_bb[3]), (0, 255, 0), 2)
                        cv2.imshow("YOLOv8 Tracking", frame)
                        cv2.waitKey(25)
                    is_first_frame = False
                    frame_count += 1
                else:
                    results = model.track(frame, classes = BALL_CLASS, conf = DEFAULT_CONFIDENCE, max_det = MAX_DET, persist=True)
                    yolo_bbox, average_bb, num_of_mock_bb = create_tuple_bbox_from_results_v2(results, average_bb, num_of_mock_bb, frame=frame, show_plot=show_plot)
                    if abs(yolo_bbox[0]-average_bb[0]) > 100 and frame_count > N / 3:
                        break
                    if (num_of_mock_bb > 20 and frame_count > N * 3 / 4) or frame_count == N - 1: # Stop tracking if the ball is not detected for 10 consecutive frames
                        prediction_bb = prediction_bb[:-num_of_mock_bb]
                        break
                    prediction_bb.append(yolo_bbox)
                    frame_count += 1  
                    if show_plot:
                        annotated_frame = results[0].plot()
                        cv2.imshow("YOLOv8 Tracking", annotated_frame)
                        # Break the loop if 'q' is pressed
                        if cv2.waitKey(25) & 0xFF == ord("q"):
                            break
            else:
                # Break the loop if the end of the video is reached
                break
        cap.release()
        query_name = f"{video_id}{_PREDICTED}{DOT_TXT}"
        file_path = os.path.join(os.getcwd(), PATH_TO_PREDICTIONS, query_name)    
        
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))  
        
        # Write the predictions to a text file
        with open(file_path, 'w') as file:
            file.write(f"{N} -1 -1 -1 -1\n")
            for idx, bbox in enumerate(prediction_bb):
                if idx == len(prediction_bb) - 1:
                    file.write(f"{idx} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}")
                else:    
                    file.write(f"{idx} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")    
        if EVAL:
            # Evaluate the predictions
            gt_file_path = f'{PATH_TO_GT}/{video_id}{_GT}{DOT_TXT}'
            _, bounding_boxes = read_bounding_boxes(gt_file_path)
            num_matches = 0
            for i in range(len(bounding_boxes)):
                if i < len(prediction_bb):
                    if compute_iou(prediction_bb[i], bounding_boxes[i]) > 0.3:
                        num_matches += 1
                else:
                    break
            print(f"Percentage of matches: {num_matches / len(bounding_boxes)}")
            dict_of_results[video_id] = num_matches / len(bounding_boxes)
    print(dict_of_results)
    return 0                

def main():
    # return_code = task2(RUNTYPE_TRAIN, SHOW_PLOT)
    # return_code = task2(RUNTYPE_FAKE_TEST, SHOW_PLOT)
    return_code = task2(RUNTYPE_TEST, True)
    if return_code == 0:
        print("Task2 completed successfully")
    elif return_code == 1:
        print("Error in loading the video")

if __name__ == "__main__":
    main()  
    