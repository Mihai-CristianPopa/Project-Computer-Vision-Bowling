import cv2
import os
from ultralytics import YOLO
from ultralytics.engine.results import Results
from typing import List, Dict, Tuple, Literal, Optional

PATH_TO_MODEL = "PinDetection.pt"
PATH_TO_TEMPLATES = "data/train/Task1/full-configuration-templates"
PATH_TO_TRAIN_IMAGES = "data/train/Task1"
PATH_TO_TRAIN_GT = "data/train/Task1/ground-truth"
PATH_TO_TRAIN_PREDICTIONS = "results/train/Task1"
PATH_TO_FAKE_TEST_IMAGES = "data/evaluation/fake_test/Task1"
PATH_TO_FAKE_TEST_GT = "data/evaluation/fake_test/ground-truth/Task1"
PATH_TO_FAKE_TEST_PREDICTIONS = "results/fake_test/Task1"
PATH_TO_TEST_IMAGES = "data/test/Task1"
PATH_TO_TEST_GT = "data/test/ground-truth/Task1"
PATH_TO_TEST_PREDICTIONS = "Mihai_Popa_407/Task1"
PATH_TO_SAVE_IMAGES = "save_img/Task1"
X_START = 280 # remove 280 pixels from the left
X_END = 1000 # remove 280 pixels from the right
IOU_THRESHOLD = 0.34
POSSIBLY_OBSTRUCTED_PINS = [4, 7, 8]
DEFAULT_CONFIDENCE = 0.55
IMG_PREDICT_CONFIDENCE = 0.07
MAX_DETECTIONS = 10
PIN_CLASS = [1]
DOT_JPG = ".jpg"
DOT_PNG = ".png"
DOT_TXT = ".txt" 
_PREDICTED = "_predicted"
OBSTRUCTED_MIDDLE_PINS = "obstructed_middle_pins"
UNBALANCED_CAMERA = "unbalanced_camera"
OBSTRUCTED = "obstructed"
UNOBSTRUCTED = "unobstructed"
SAVE = "save"
DISPLAY = "display"
LEFT = "left"
RIGHT = "right"
TOP = "top"
BOT = "bot"

RUNTYPE_TRAIN = "train"
RUNTYPE_FAKE_TEST = "fake_test"
RUNTYPE_TEST = "test"

Runtype = Literal["train", "fake_test", "test"]
DisplayMode = Optional[Literal["save", "display"]]
Mode = Literal["save", "display"]
ObstructedBox = Literal["obstructed", "unobstructed"]

def is_file_of_type(filename: str, ending: str) -> bool:
    return filename.endswith(ending)

def load_images(directory: str, crop=False, resize=False, show=False) -> list:

    images = []
    try:
        for filename in os.listdir(directory):
            if is_file_of_type(filename, DOT_JPG) or is_file_of_type(filename, DOT_PNG):
                img = cv2.imread(os.path.join(directory, filename))
                if img is not None:
                    if crop: # making the images squares
                        img = img[:, X_START : X_END]
                    if resize:
                        img = cv2.resize(img, (640, 640)) # resizing to match the training data
                    if show:
                        display_image(img)    
                    images.append(img)
    except FileNotFoundError:
        return None
    return images

def load_queries(directory: str) -> list:

    queries = []
    try:
        for filename in os.listdir(directory):
            if is_file_of_type(filename, DOT_TXT):
                with open(os.path.join(directory, filename), 'r') as file:
                    query = [int(line.strip()) for line in file]
                queries.append((query[0],query[1:]))
    except FileNotFoundError:
        return None            
    return queries

def display_image(image: cv2.typing.MatLike):

    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_bounding_boxes(image: cv2.typing.MatLike, bounding_boxes: List[Dict], obstructedBox: ObstructedBox=OBSTRUCTED) -> cv2.typing.MatLike:
    
    for idx, box in enumerate(bounding_boxes):
        if obstructedBox in box:
            box = box[obstructedBox]
        left, top, right, bottom = int(box[LEFT]), int(box[TOP]), int(box[RIGHT]), int(box[BOT])
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, str(idx), (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def display_or_save_images_with_bounding_boxes(image_list: List[cv2.typing.MatLike], all_images_pins_bounding_boxes: List[List[Dict]], mode: Mode=DISPLAY, save_dir="save-img/Task1", obstructedBox: ObstructedBox=OBSTRUCTED):
    
    if mode == SAVE and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for i, image in enumerate(image_list):
        bounding_boxes = all_images_pins_bounding_boxes[i]
        image_with_boxes = draw_bounding_boxes(image, bounding_boxes, obstructedBox)
        
        if mode == DISPLAY:
            cv2.imshow(f'Image {i+1}', image_with_boxes)
        elif mode == SAVE:
            if "lane" in save_dir:
                image_name = f"lane_{i+1}.png"
            else:
                image_name = f"0{i+1}.png" if i + 1 < 10 else f"{i+1}.png" 
            save_path = os.path.join(save_dir, image_name)
            cv2.imwrite(save_path, image_with_boxes)
    
    if mode == DISPLAY:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def get_max_intensity(image: cv2.typing.MatLike, lane_image: cv2.typing.MatLike) -> float:
    
    result = cv2.matchTemplate(image, lane_image, cv2.TM_CCOEFF_NORMED)
    
    _, max_val, _, _ = cv2.minMaxLoc(result)
    
    return max_val

def get_index_of_best_fitting_lane(image: cv2.typing.MatLike, lane_img_list: List[cv2.typing.MatLike]) -> int:

    max_val_list = [get_max_intensity(image, lane_image) for lane_image in lane_img_list]

    return max_val_list.index(max(max_val_list))

def get_results_for_an_image(model: YOLO, image: cv2.typing.MatLike, classes=PIN_CLASS, conf=DEFAULT_CONFIDENCE, max_det=MAX_DETECTIONS) -> Results:
    results = model.predict(image, classes=classes, conf=conf, max_det=max_det)
    return results[0]

def build_unobstructed_dict(bb: dict) -> dict:
    return {LEFT: bb[LEFT], RIGHT: bb[RIGHT], TOP: bb[TOP], BOT: bb[BOT]}

def compute_middle_point(bb1: dict, bb2: dict, bb3: dict=None, bb4: dict =None) -> dict:
    bb = {}
    if bb3 and bb4: # if we are computing the bb for the pins from the last line
        bb[LEFT] = (bb3[LEFT] + bb4[LEFT]) / 2
        bb[RIGHT] = (bb3[RIGHT] + bb4[RIGHT]) / 2
    else:    
        bb[LEFT] = (bb1[LEFT] + bb2[LEFT]) / 2
        bb[RIGHT] = (bb1[RIGHT] + bb2[RIGHT]) / 2
    bb[TOP] = (bb1[TOP] + bb2[TOP]) / 2
    bb[BOT] = (bb1[BOT] + bb2[BOT]) / 2
    return bb

def update_bottom_value_with_line_average(indices: list, all_sorted_list_bb: List[List]):
    new_bottom_value =  sum(all_sorted_list_bb[i][3] for i in indices) / len(indices)
    for i in indices:
        all_sorted_list_bb[i][3] = new_bottom_value

def get_bounding_boxes_from_results(results: Results, template_issue="") -> List[Dict]:
    # Getting the list of bounding boxes
    all_pins_bounding_box_list = []
    xyxys = results.boxes.xyxy.cpu().numpy()
    for pin_bounding_box in xyxys:
        all_pins_bounding_box_list.append(pin_bounding_box.tolist())
    
    # Sorting the bounding boxes descending by the bottom value and ascending by the left value
    all_sorted_list_bb =  sorted(all_pins_bounding_box_list, key=lambda x: [-x[3], x[0]])
    bounding_boxes_dicts = [{LEFT: bb[0], RIGHT: bb[2], TOP: bb[1], BOT: bb[3]} for bb in all_sorted_list_bb]

    # There were two categories of edge cases for the templates bounding boxes, that caused poor sorting of the bounding boxes
    if template_issue == OBSTRUCTED_MIDDLE_PINS:
        # The middle pins are obstructed so we construct bounding boxes based on the side pins
        # by averaging the coordinates from the side pins
        pin_number_5 = {
            UNOBSTRUCTED: build_unobstructed_dict(
                            compute_middle_point(bounding_boxes_dicts[3], 
                                                 bounding_boxes_dicts[4])),
            OBSTRUCTED: bounding_boxes_dicts[9]
                        }
        # For the pins from the last line we compute taking into consideration the sides of the pins
        # in front of them. While for top bot coordinates we use the side pins from the same line
        pin_number_8 = {
            UNOBSTRUCTED: build_unobstructed_dict(
                            compute_middle_point(bounding_boxes_dicts[5],
                                                 bounding_boxes_dicts[6],
                                                 bounding_boxes_dicts[3],
                                                 pin_number_5[UNOBSTRUCTED])),
            OBSTRUCTED: bounding_boxes_dicts[8]
        }

        pin_number_9 = {
            UNOBSTRUCTED: build_unobstructed_dict(
                            compute_middle_point(bounding_boxes_dicts[5],
                                                 bounding_boxes_dicts[6],
                                                 bounding_boxes_dicts[4],
                                                 pin_number_5[UNOBSTRUCTED])),
            OBSTRUCTED: bounding_boxes_dicts[7]
        }

        final_bounding_boxes_dicts = (
            bounding_boxes_dicts[:4] +
            [pin_number_5] +
            bounding_boxes_dicts[4:6] +
            [pin_number_8, pin_number_9] +
            [bounding_boxes_dicts[6]]
        )
        bounding_boxes_dicts = final_bounding_boxes_dicts
    # For two of the templates, there were lines where the bottom of the bounding boxes
    # were not aligned, so we had to adjust them 
    elif template_issue == UNBALANCED_CAMERA:
        # Calculate new bottom values for each line
        update_bottom_value_with_line_average([1, 2], all_sorted_list_bb)
        update_bottom_value_with_line_average([3, 4, 5], all_sorted_list_bb)
        update_bottom_value_with_line_average([6, 7, 8, 9], all_sorted_list_bb)
        final_sorted_bb_list =  sorted(all_sorted_list_bb, key=lambda x: [-x[3], x[0]])
        bounding_boxes_dicts = [{LEFT: bb[0], RIGHT: bb[2], TOP: bb[1], BOT: bb[3]} for bb in final_sorted_bb_list]
    return bounding_boxes_dicts

                        
def merge_multiple_image_pin_bounding_boxes(model: YOLO, image_list: List[cv2.typing.MatLike], conf=0.55, template_issue="") -> List[List[Dict]]:
    # Getting the bounding boxes for all images from the list
    all_images_pins_bounding_boxes = []
    for i in range(len(image_list)):
        all_images_pins_bounding_boxes.append(get_bounding_boxes_from_results(
            get_results_for_an_image(model, image_list[i], conf=conf),
            template_issue
        ))
    return all_images_pins_bounding_boxes    

def is_slightly_translated(bb1: dict, bb2: dict) -> bool:
    # There were some cases where the side points of the bounding boxes were slightly translated
    # and that made the intersection area to be 0, so we had to check if the bounding boxes are slightly translated
    # whilst the top and bottom points being pretty close
    return abs(bb1[LEFT] - bb2[LEFT]) < 65 and abs(bb1[RIGHT] - bb2[RIGHT]) < 65 and abs(bb1[TOP] - bb2[TOP]) < 10 and abs(bb1[BOT] - bb2[BOT]) < 20

def compute_iou(bb1: dict, bb2: dict) -> float:

    assert bb1[LEFT] < bb1[RIGHT]
    assert bb1[TOP] < bb1[BOT]
    assert bb2[LEFT] < bb2[RIGHT]
    assert bb2[TOP] < bb2[BOT]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[LEFT], bb2[LEFT])
    y_top = max(bb1[TOP], bb2[TOP])
    x_right = min(bb1[RIGHT], bb2[RIGHT])
    y_bottom = min(bb1[BOT], bb2[BOT])

    # there were some cases where there was an almost full overlap between the left and right points
    # but the bottom variation was significant which implied that it might be an overlap with the bounding box
    # of a pin from the line in front
    # In that case I wanted to return an IoU of 0
    if abs(bb1[BOT]-bb2[BOT]) > 30:
        return 0.0

    # check if there is an intersection
    if (x_right < x_left or y_bottom < y_top):
        # there is no intersection, but top and bottom are pretty close
        # then take the middle point of the bounding boxes and compute the intersection area
        if is_slightly_translated(bb1, bb2):
            x_left = (bb1[LEFT] + bb2[LEFT]) / 2
            x_right = (bb1[RIGHT] + bb2[RIGHT]) / 2
        else:
            return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[RIGHT] - bb1[LEFT]) * (bb1[BOT] - bb1[TOP])
    bb2_area = (bb2[RIGHT] - bb2[LEFT]) * (bb2[BOT] - bb2[TOP])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou      

def match_bb_based_on_key(train_boxes: List[Dict], template_box_based_on_key: dict, key: int, iou_threshold=IOU_THRESHOLD) -> Tuple[bool, dict]:
    # Check each bounding box from the image used for prediction
    for train_box in train_boxes:
        # If the template bounding box is one that contains both obstructed and unobstructed pins, check the IoU with both
        if key in POSSIBLY_OBSTRUCTED_PINS and template_box_based_on_key.get(UNOBSTRUCTED):
            iou = max(compute_iou(train_box, template_box_based_on_key[UNOBSTRUCTED]),
                    compute_iou(train_box, template_box_based_on_key[OBSTRUCTED]))
        else:
            iou = compute_iou(train_box, template_box_based_on_key)    
        if iou > iou_threshold:
            return True, train_box
    return False, None

def load_results(directory: str) -> list:
    queries = []
    for filename in os.listdir(directory):
        if is_file_of_type(filename, DOT_TXT):
            with open(os.path.join(directory, filename), 'r') as file:
                query = [line.strip() for line in file]
            queries.append(" | ".join(query[1:]))
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

def task1(runtype: Runtype=RUNTYPE_TRAIN, display_mode: DisplayMode=None):

    yolo_model = YOLO(PATH_TO_MODEL)

    if runtype == RUNTYPE_TRAIN:
        PATH_TO_QUERIES = PATH_TO_TRAIN_IMAGES
        PATH_TO_IMAGES = PATH_TO_TRAIN_IMAGES
        PATH_TO_SAVE_LANE_IMAGES = PATH_TO_SAVE_IMAGES + "/train/lane_images"
        PATH_TO_SAVE_PREDICTED_IMAGES = PATH_TO_SAVE_IMAGES + "/train/predicted_images"
        PATH_TO_PREDICTIONS = PATH_TO_TRAIN_PREDICTIONS
        PATH_TO_GT = PATH_TO_TRAIN_GT
        EVAL = True
    elif runtype == RUNTYPE_FAKE_TEST:
        PATH_TO_QUERIES = PATH_TO_FAKE_TEST_IMAGES
        PATH_TO_IMAGES = PATH_TO_FAKE_TEST_IMAGES
        PATH_TO_SAVE_LANE_IMAGES = PATH_TO_SAVE_IMAGES + "/fake_test/lane_images"
        PATH_TO_SAVE_PREDICTED_IMAGES = PATH_TO_SAVE_IMAGES + "/fake_test/predicted_images"
        PATH_TO_PREDICTIONS = PATH_TO_FAKE_TEST_PREDICTIONS
        PATH_TO_GT = PATH_TO_FAKE_TEST_GT
        EVAL = True
    elif runtype == RUNTYPE_TEST:
        PATH_TO_QUERIES = PATH_TO_TEST_IMAGES
        PATH_TO_IMAGES = PATH_TO_TEST_IMAGES
        PATH_TO_SAVE_LANE_IMAGES = PATH_TO_SAVE_IMAGES + "/test/lane_images"
        PATH_TO_SAVE_PREDICTED_IMAGES = PATH_TO_SAVE_IMAGES + "/test/predicted_images"
        PATH_TO_PREDICTIONS = PATH_TO_TEST_PREDICTIONS
        PATH_TO_GT = PATH_TO_TEST_GT
        EVAL = False

    lane_img_list = load_images(PATH_TO_TEMPLATES, crop=True, resize=True)
    query_list = load_queries(PATH_TO_QUERIES)
    img_to_predict_list = load_images(PATH_TO_IMAGES, crop=True, resize=True)

    if (lane_img_list is None) or (query_list is None) or (img_to_predict_list is None):
        return 1
    
    # Split the lane images as there are two groups of edge cases
    lane_img_obstructed_list = lane_img_list[:2]
    lane_img_obstructed_bb_list = merge_multiple_image_pin_bounding_boxes(yolo_model, lane_img_obstructed_list, template_issue = OBSTRUCTED_MIDDLE_PINS)
    if display_mode:
        display_or_save_images_with_bounding_boxes(lane_img_obstructed_list, lane_img_obstructed_bb_list, mode=display_mode, save_dir=f"{PATH_TO_SAVE_LANE_IMAGES}/obstructed")
        display_or_save_images_with_bounding_boxes(load_images(PATH_TO_TEMPLATES, crop=True, resize=True)[:2], lane_img_obstructed_bb_list, mode=display_mode, save_dir=f"{PATH_TO_SAVE_LANE_IMAGES}/unobstructed", obstructedBox=UNOBSTRUCTED)

    lane_img_unbalanced_cam_list = lane_img_list[2:]
    lane_img_unbalanced_cam_bb_list = merge_multiple_image_pin_bounding_boxes(yolo_model, lane_img_unbalanced_cam_list, template_issue = UNBALANCED_CAMERA)
    if display_mode:
        display_or_save_images_with_bounding_boxes(lane_img_unbalanced_cam_list, lane_img_unbalanced_cam_bb_list, mode=display_mode, save_dir=f"{PATH_TO_SAVE_LANE_IMAGES}/unbalanced")

    # Get the bounding boxes from the lane images
    lane_img_bb_list = lane_img_obstructed_bb_list + lane_img_unbalanced_cam_bb_list

    # Get the bounding boxes from the images to predict
    img_to_predict_bb_list = merge_multiple_image_pin_bounding_boxes(yolo_model, img_to_predict_list, conf=IMG_PREDICT_CONFIDENCE)
    if display_mode:
        display_or_save_images_with_bounding_boxes(img_to_predict_list, img_to_predict_bb_list, mode=display_mode, save_dir=PATH_TO_SAVE_PREDICTED_IMAGES)

    # For each image to predict find out what is the matching lane image index
    lane_img_index_list = [get_index_of_best_fitting_lane(img_to_predict, lane_img_list) for img_to_predict in img_to_predict_list]

    for i, lane_img_index in enumerate(lane_img_index_list):
        
        query_responses_dict = {}
        current_query = query_list[i]
        # Go through the keys in reverse order to match the pins from the last line first
        for key in current_query[1][::-1]: 
            isMatch, train_bb_to_remove = match_bb_based_on_key(img_to_predict_bb_list[i], lane_img_bb_list[lane_img_index][key - 1], key - 1)
            if isMatch:
                pin_state = 1
                img_to_predict_bb_list[i].remove(train_bb_to_remove)
            else:
                pin_state = 0
            query_responses_dict[key] = pin_state
        # Write the results to a file
        query_name = f"0{i+1}{_PREDICTED}{DOT_TXT}" if i + 1 <10 else f"{i+1}{_PREDICTED}{DOT_TXT}"
            
        file_path = os.path.join(os.getcwd(), PATH_TO_PREDICTIONS, query_name)
        
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        
        with open(file_path, 'w') as file:
            num_predicted_pins = current_query[0]
            file.write(f"{num_predicted_pins}\n")
            for key, value in list(query_responses_dict.items())[::-1]:
                if num_predicted_pins > 1:
                    file.write(f"{key} {value}\n")
                else:
                    file.write(f"{key} {value}")
                num_predicted_pins -= 1    

    if EVAL:    
        predicted_truths = load_results(PATH_TO_PREDICTIONS)
        ground_truths = load_results(PATH_TO_GT)
        acc = calculate_accuracy(predicted_truths, ground_truths)
        print(acc)
    return 0

def main():
    # return_code = task1(RUNTYPE_TRAIN, SAVE)
    # return_code = task1(RUNTYPE_FAKE_TEST, SAVE)
    return_code = task1(RUNTYPE_TEST, SAVE)
    if return_code == 0:
        print("Task1 completed successfully")
    elif return_code == 1:
        print("Error in loading the data")

if __name__ == "__main__":
    main()               