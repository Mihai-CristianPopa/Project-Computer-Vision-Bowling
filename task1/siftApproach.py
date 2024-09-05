
# # Important NOTE:  Use opencv >=4.4 
# import cv2
 
# # Loading the image
# # img = cv2.imread('data/train/Task1/full-configuration-templates/lane1.jpg')
# img = cv2.imread('data/train/Task1/01.jpg')
 
#  # Converting image to grayscale
# gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# # Applying SIFT detector
# sift = cv2.SIFT_create()
# kp = sift.detect(gray, None)
 
# # Marking the keypoint on the image using circles
# img=cv2.drawKeypoints(gray ,
#                       kp ,
#                       img ,
#                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# cv2.imwrite('results/image-with-keypoints.jpg', img)

# -------------V2

import cv2
import numpy as np
import os
from ultralytics import YOLO

PATH_TO_TEMPLATES = "data/train/Task1/full-configuration-templates"
PATH_TO_TRAIN_IMAGES = "data/train/Task1"
PATH_TO_RESULTS = "results/Task1/segmentation"
X_START = 280 # remove 280 pixels from the left
X_END = 1000 # remove 280 pixels from the right
POSSIBLY_OBSTRUCTED_PINS = [4, 7, 8] 

# def is_dict_of_dicts(entry):
#     if isinstance(entry, dict):
#         return all(isinstance(value, dict) for value in entry.values())
#     return False

def load_images(directory, crop=False, resize=False, show=False) -> list:
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(directory, filename))
            if img is not None:
                if crop: # making the images squares
                    img = img[:, X_START : X_END]
                if resize:
                    img = cv2.resize(img, (640, 640)) # resizing to match the training data
                if show:
                    display_image(img)    
                images.append(img)
    return images

def load_queries(directory):
    queries = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r') as file:
                # Read all lines and strip newline characters
                query = [int(line.strip()) for line in file]
            queries.append((query[0],query[1:]))
    return queries

def display_image(image):
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_bounding_boxes(image, bounding_boxes, obstructedBox="obstructed"):
    for idx, box in enumerate(bounding_boxes):
        if obstructedBox in box:
            box = box[obstructedBox]
        left, top, right, bottom = int(box['left']), int(box['top']), int(box['right']), int(box['bot'])            
        # left, top, right, bottom = map(int, box)
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, str(idx), (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def display_or_save_images_with_bounding_boxes(image_list, all_images_pins_bounding_boxes, mode='display', save_dir='results/Task1/bounding-boxes', obstructedBox="obstructed"):
    if mode == 'save' and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for i, image in enumerate(image_list):
        bounding_boxes = all_images_pins_bounding_boxes[i]
        image_with_boxes = draw_bounding_boxes(image, bounding_boxes, obstructedBox)
        
        if mode == 'display':
            cv2.imshow(f'Image {i+1}', image_with_boxes)
        elif mode == 'save':
            save_path = os.path.join(save_dir, f'image_{i+1}.png')
            cv2.imwrite(save_path, image_with_boxes)
    
    if mode == 'display':
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def get_max_intensity(image, lane_image):
    # Apply template matching
    result = cv2.matchTemplate(image, lane_image, cv2.TM_CCOEFF_NORMED)
    # Get the best match position
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val

def get_index_of_best_fitting_lane(image, lane_image_list) -> int:

    max_val_list = [get_max_intensity(image, lane_image) for lane_image in lane_image_list]

    return max_val_list.index(max(max_val_list))

def get_results_for_an_image(model, image, classes=[1], conf=0.55, max_det=10):
    results = model.predict(image, classes=classes, conf=conf, max_det=max_det)
    return results[0]

def get_bounding_boxes_from_results(results, obstructed_middle_pins=""):
    all_pins_bounding_box_list = []
    xyxys = results.boxes.xyxy.cpu().numpy()
    for pin_bounding_box in xyxys:
        all_pins_bounding_box_list.append(pin_bounding_box.tolist())
    # return sorted(all_pins_bounding_box_list, key=lambda x: [x[3], x[0]], reverse=True)

    # all_sorted_list_bb =  sorted(all_pins_bounding_box_list, key=lambda x: [x[3], x[0]], reverse=True)
    # bounding_boxes_dicts = [{'left': bb[0], 'right': bb[2], 'top': bb[1], 'bot': bb[3]} for bb in all_sorted_list_bb]
    # return bounding_boxes_dicts
    all_sorted_list_bb =  sorted(all_pins_bounding_box_list, key=lambda x: [-x[3], x[0]])
    bounding_boxes_dicts = [{'left': bb[0], 'right': bb[2], 'top': bb[1], 'bot': bb[3]} for bb in all_sorted_list_bb]
    if obstructed_middle_pins == "obstructed_middle_pins":
        pin_number_5 = {
            "unobstructed": {
                'left': (all_sorted_list_bb[3][0] + all_sorted_list_bb[4][0]) / 2, 
                'right': (all_sorted_list_bb[3][2] + all_sorted_list_bb[4][2]) / 2, 
                'top': (all_sorted_list_bb[3][1] + all_sorted_list_bb[4][1]) / 2, 
                'bot': (all_sorted_list_bb[3][3] + all_sorted_list_bb[4][3]) / 2},
            "obstructed": bounding_boxes_dicts[9]
                        }
        # bounding_boxes_dicts[4], bounding_boxes_dicts[5] = pin_number_5, bounding_boxes_dicts[4]
        pin_number_8 = {
            "unobstructed": {
                'left': (all_sorted_list_bb[3][0] + pin_number_5["unobstructed"]["left"]) / 2, 
                'right': (all_sorted_list_bb[3][2] + pin_number_5["unobstructed"]["right"]) / 2, 
                'top': (all_sorted_list_bb[5][1] + all_sorted_list_bb[6][1]) / 2, 
                'bot': (all_sorted_list_bb[5][3] + all_sorted_list_bb[6][3]) / 2},
            "obstructed": bounding_boxes_dicts[8]
        }
        # bounding_boxes_dicts[8] = pin_number_8
        pin_number_9 = {
            "unobstructed": {
                'left': (all_sorted_list_bb[4][0] + pin_number_5["unobstructed"]["left"]) / 2, 
                'right': (all_sorted_list_bb[4][2] + pin_number_5["unobstructed"]["right"]) / 2,
                'top': (all_sorted_list_bb[5][1] + all_sorted_list_bb[6][1]) / 2, 
                'bot': (all_sorted_list_bb[5][3] + all_sorted_list_bb[6][3]) / 2},
            "obstructed": bounding_boxes_dicts[7]
        }
        final_bounding_boxes_dicts = bounding_boxes_dicts[:4]
        final_bounding_boxes_dicts.append(pin_number_5)
        final_bounding_boxes_dicts.extend(bounding_boxes_dicts[4:6])
        final_bounding_boxes_dicts.append(pin_number_8)
        final_bounding_boxes_dicts.append(pin_number_9)
        final_bounding_boxes_dicts.append(bounding_boxes_dicts[6])
        # bounding_boxes_dicts = bounding_boxes_dicts[:4] + pin_number_5 + bounding_boxes_dicts[4:6] + pin_number_8 + pin_number_9 + bounding_boxes_dicts[6]
        bounding_boxes_dicts = final_bounding_boxes_dicts
    elif obstructed_middle_pins == "unbalanced_camera":
        new_bottom_value_line1 = (all_sorted_list_bb[1][3] + all_sorted_list_bb[2][3]) / 2
        new_bottom_value_line2 = (all_sorted_list_bb[3][3] + all_sorted_list_bb[4][3] + all_sorted_list_bb[5][3]) / 3
        new_bottom_value_line3 = (all_sorted_list_bb[6][3] + all_sorted_list_bb[7][3] + all_sorted_list_bb[8][3] + all_sorted_list_bb[9][3]) / 4
        all_sorted_list_bb[1][3] = new_bottom_value_line1
        all_sorted_list_bb[2][3] = new_bottom_value_line1
        all_sorted_list_bb[3][3] = new_bottom_value_line2
        all_sorted_list_bb[4][3] = new_bottom_value_line2
        all_sorted_list_bb[5][3] = new_bottom_value_line2
        all_sorted_list_bb[6][3] = new_bottom_value_line3
        all_sorted_list_bb[7][3] = new_bottom_value_line3
        all_sorted_list_bb[8][3] = new_bottom_value_line3
        all_sorted_list_bb[9][3] = new_bottom_value_line3
        final_sorted_bb_list =  sorted(all_sorted_list_bb, key=lambda x: [-x[3], x[0]])
        bounding_boxes_dicts = [{'left': bb[0], 'right': bb[2], 'top': bb[1], 'bot': bb[3]} for bb in final_sorted_bb_list]
    return bounding_boxes_dicts

                        
def merge_multiple_image_pin_bounding_boxes(model, image_list, conf=0.55, obstructed_middle_pins=""):
    all_images_pins_bounding_boxes = []
    for i in range(len(image_list)):
        all_images_pins_bounding_boxes.append(get_bounding_boxes_from_results(
            get_results_for_an_image(model, image_list[i], conf=conf),
            obstructed_middle_pins
        ))
    return all_images_pins_bounding_boxes    

def is_slightly_translated(bb1, bb2):
    return abs(bb1['left'] - bb2['left']) < 65 and abs(bb1['right'] - bb2['right']) < 65 and abs(bb1['top'] - bb2['top']) < 10 and abs(bb1['bot'] - bb2['bot']) < 20

# def calculate_point_closeness(bb1_point, bb2_point):
#      return (abs(bb1_point - bb2_point) + 1) / ((bb1_point + bb2_point) / 2 + 1)

def compute_iou(bb1, bb2) -> float:
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'left', 'right', 'top', 'bot'}
        The (left, top) position is at the top left corner,
        the (right, bot) position is at the bottom right corner
    bb2 : dict
        Keys: {'left', 'right', 'top', 'bot'}
        The (x, y) position is at the top left corner,
        the (right, bot) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['left'] < bb1['right']
    assert bb1['top'] < bb1['bot']
    assert bb2['left'] < bb2['right']
    assert bb2['top'] < bb2['bot']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['left'], bb2['left'])
    y_top = max(bb1['top'], bb2['top'])
    x_right = min(bb1['right'], bb2['right'])
    y_bottom = min(bb1['bot'], bb2['bot'])

    if abs(bb1['bot']-bb2['bot']) > 30:
        return 0.0

    # abs(bb1['left'] - bb2['left']) < 65
    # abs(bb1['right'] - bb2['right']) < 65
    # abs(bb1['top'] - bb2['top']) < 10
    # abs(bb1['bot'] - bb2['bot']) < 20

    if (x_right < x_left or y_bottom < y_top):
        if is_slightly_translated(bb1, bb2):
            x_left = (bb1['left'] + bb2['left']) / 2
            x_right = (bb1['right'] + bb2['right']) / 2
        else:
            return 0.0

    # Calculate the closeness of top and bot values
    # top_closeness = calculate_point_closeness(bb1['top'], bb2['top'])
    # bot_closeness = calculate_point_closeness(bb1['bot'], bb2['bot'])
    # top_bot_closeness = 1 - top_closeness * bot_closeness


    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['right'] - bb1['left']) * (bb1['bot'] - bb1['top'])
    bb2_area = (bb2['right'] - bb2['left']) * (bb2['bot'] - bb2['top'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    # weighted_iou = (1 - w) * iou + w * top_bot_closeness
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def match_bounding_boxes(train_boxes, template_boxes, iou_threshold=0.5):
    # {idx_train_box: {
    # }}
    matched_boxes = {} # make it a dictionary with the key the pin id
                        # so that you can query the dictionary to get the pin id
    # Start from the last matched template box, because we are taking the pins in order
    last_matched_template_box = 0
    for idx_train, train_box in enumerate(train_boxes):
        # for i, template_box in enumerate(template_boxes):
        for i in range(last_matched_template_box, len(template_boxes)):
            if i in POSSIBLY_OBSTRUCTED_PINS and template_boxes[i].get("unobstructed"):
                iou = max(compute_iou(train_box, template_boxes[i]["unobstructed"]),
                        compute_iou(train_box, template_boxes[i]["obstructed"]))
            else:
                iou = compute_iou(train_box, template_boxes[i])    
            if iou > iou_threshold:
                last_matched_template_box = i
                matched_boxes[i] = {
                    "train_index_box": idx_train,
                    "pin_position_indexed_from_1": i + 1,
                    "iou": iou
                }
                break
    return matched_boxes        

def match_based_bb_based_on_key(train_boxes, template_box_based_on_key, key, iou_threshold=0.34):
    # if key in POSSIBLY_OBSTRUCTED_PINS and template_box_based_on_key.get("unobstructed"):
    #             iou = max(compute_iou(train_box, template_boxes[i]["unobstructed"]),
    #                     compute_iou(train_box, template_boxes[i]["obstructed"]))
    # else:
    for train_box in train_boxes:
        if key in POSSIBLY_OBSTRUCTED_PINS and template_box_based_on_key.get("unobstructed"):
            iou = max(compute_iou(train_box, template_box_based_on_key["unobstructed"]),
                    compute_iou(train_box, template_box_based_on_key["obstructed"]))
        else:
            iou = compute_iou(train_box, template_box_based_on_key)    
        if iou > iou_threshold:
            return True, train_box
    return False, None

def load_results(directory):
    queries = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r') as file:
                query = [line.strip() for line in file]
            queries.append(" | ".join(query[1:]))
    return queries

def calculate_accuracy(predicted_truths, ground_truths):
    sum = 0
    index = 1
    for prediction,truth in zip(predicted_truths, ground_truths):
        if prediction == truth:
            sum += 1
        else:
            print(index)
        index += 1        
    return sum / len(predicted_truths)

# [:, x_start : x_end]

# 1. Load images
# img2 = cv2.imread('data/train/Task1/01.jpg')   # Image after hitting the pins
# lane_image_list = load_images(PATH_TO_TEMPLATES)
# index = get_index_of_best_fitting_lane(img2, lane_image_list)
# matching_lane_image = lane_image_list[index]

# model = YOLO("yolov8x.pt")
# model.predict(matching_lane_image, show=True)
# model.predict(img2, show=True)

# model2 = YOLO("yolov10x.pt")
# model2.predict(matching_lane_image, show=True)
# model2.predict(img2, show=True)

# model3 = YOLO("yolov8x-seg.pt")
# model3.predict(matching_lane_image, show=True)
# result = model3.predict(img2, show=True)
# results = model3.predict(PATH_TO_TEMPLATES)
# for i in range(len(results)):
#     results[i].save(PATH_TO_RESULTS + f"cropped/lane{i+1}.jpg")

# image_segmentation_list = model3.predict(PATH_TO_TRAIN_IMAGES)
# for i in range(len(image_segmentation_list)):
#     image_segmentation_list[i].save(PATH_TO_RESULTS + f"cropped/{i+1}.jpg")
    
# cropped_train_image_list = load_images(PATH_TO_TRAIN_IMAGES, crop=True)      
# cropped_template_image_list = load_images(PATH_TO_TEMPLATES, crop=True)
# for i in range(len(cropped_template_image_list)):
#     s = model3.predict(cropped_template_image_list[i])
#     s.save(PATH_TO_RESULTS + f"cropped/lane{i+1}.jpg")
# for i in range(len(cropped_train_image_list)):
#     s = model3.predict(cropped_train_image_list[i])
#     s.save(PATH_TO_RESULTS + f"cropped/{i+1}.jpg")

model4 = YOLO("PinDetection.pt")
# model4.predict('data/train/Task1/25.jpg', classes=[1], conf=0.07, max_det=10)[0].save('results/Task1/segmentation/improvements/25.jpg')
query_list = load_queries(PATH_TO_TRAIN_IMAGES)
template_images = load_images(PATH_TO_TEMPLATES, crop=True, resize=True)
train_images = load_images(PATH_TO_TRAIN_IMAGES, crop=True, resize=True)

# Split the lane images as there are two groups of edge cases
lane_image_obstructed_middle_pins = template_images[:2]
template_issue = "obstructed_middle_pins"
lane_image_obstructed_middle_pins_bounding_box = merge_multiple_image_pin_bounding_boxes(model4, lane_image_obstructed_middle_pins, obstructed_middle_pins = "obstructed_middle_pins")
display_or_save_images_with_bounding_boxes(lane_image_obstructed_middle_pins, lane_image_obstructed_middle_pins_bounding_box, mode='save', save_dir='results/Task1/bounding-boxes/templates/obstructed')
display_or_save_images_with_bounding_boxes(load_images(PATH_TO_TEMPLATES, crop=True, resize=True)[:2], lane_image_obstructed_middle_pins_bounding_box, mode='save', save_dir='results/Task1/bounding-boxes/templates/unobstructed', obstructedBox="unobstructed")

template_issue = "unbalanced_camera"
lane_image_unbalanced_camera = template_images[2:]
lane_image_unbalanced_camera_bounding_box = merge_multiple_image_pin_bounding_boxes(model4, lane_image_unbalanced_camera, obstructed_middle_pins = "unbalanced_camera")
display_or_save_images_with_bounding_boxes(lane_image_unbalanced_camera, lane_image_unbalanced_camera_bounding_box, mode='save', save_dir='results/Task1/bounding-boxes/templates/unbalanced')
# # Get the bounding boxes from the lane images
template_image_bounding_box_list = lane_image_obstructed_middle_pins_bounding_box + lane_image_unbalanced_camera_bounding_box

# # Get the bounding boxes from the train images
train_image_bounding_box_list = merge_multiple_image_pin_bounding_boxes(model4, train_images, conf=0.07) #conf=0.05
display_or_save_images_with_bounding_boxes(train_images, train_image_bounding_box_list, mode='save', save_dir='results/Task1/segmentation/improvements')

# For each train image
#   Find out what is the matching lane image
#       Compare the bounding boxes of the train image
#        with the bounding boxes of the matching lane image
#        is the matching lane
lane_image_index_list = [get_index_of_best_fitting_lane(train_image, template_images) for train_image in train_images]
# matches = []
query_responses = []
for i, lane_image_index in enumerate(lane_image_index_list):
    # print(3)
    query_responses_maping = {}
    current_query = query_list[i]
    # matches = match_bounding_boxes(train_image_bounding_box_list[i], template_image_bounding_box_list[lane_image_index])
    for key in current_query[1][::-1]: 
        isMatch, train_bb_to_remove = match_based_bb_based_on_key(train_image_bounding_box_list[i], template_image_bounding_box_list[lane_image_index][key - 1], key - 1)
        if isMatch:
            pin_state = 1
            train_image_bounding_box_list[i].remove(train_bb_to_remove)
        else:
            pin_state = 0
        query_responses_maping[key] = pin_state

    if i + 1 <10:
        file_path = os.path.join(f"{os.getcwd()}\\results\\train-predictions", f"0{i+1}_predicted.txt")
    else:
        file_path = os.path.join(f"{os.getcwd()}\\results\\train-predictions", f"{i+1}_predicted.txt")    
    
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    
    with open(file_path, 'w') as file:
        num_predicted_pins = current_query[0]
        file.write(f"{num_predicted_pins}\n")
        for key, value in list(query_responses_maping.items())[::-1]:
            if num_predicted_pins > 1:
                file.write(f"{key} {value}\n")
            else:
                file.write(f"{key} {value}")
            num_predicted_pins -= 1    
    # query_responses.append(query_responses_maping)
predicted_truths = load_results("results/train-predictions")
ground_truths = load_results("data/train/Task1/ground-truth")

acc = calculate_accuracy(predicted_truths, ground_truths)
print(acc)


# for i in range(len(template_images)):
#     results = model4.predict(template_images[i], classes=[1], conf=0.55, max_det=10)
#     results[0].save(PATH_TO_RESULTS + f"/cropped/lane{i+1}.jpg")
# for i in range(len(train_images)):
#     results = model4.predict(train_images[i], classes=[1], max_det=10)
#     results[0].save(PATH_TO_RESULTS + f"/cropped/{i+1}.jpg")
# from roboflow import Roboflow
# rf = Roboflow(api_key="wmirxsZw9MLKvQTTwTFB")
# project = rf.workspace("lsc-kik8c").project("bowling-pin-detection")
# version = project.version(4)
# dataset = version.download("yolov8")

# # 2. Convert to grayscale
# gray1 = cv2.cvtColor(matching_lane_image[:, x_start : x_end], cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(img2[:, x_start : x_end], cv2.COLOR_BGR2GRAY)

# display_image(gray1)
# display_image(gray2)

# edges1 = cv2.Canny(gray1, 100, 200)
# edges2 = cv2.Canny(gray2, 100, 200)

# display_image(edges1)
# display_image(edges2)

# # 3. Initialize SIFT detector
# sift = cv2.SIFT_create(contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)

# # 4. Detect keypoints and compute descriptors
# # kp1, des1 = sift.detectAndCompute(gray1, None)
# # kp2, des2 = sift.detectAndCompute(gray2, None)
# kp1, des1 = sift.detectAndCompute(edges1, None)
# kp2, des2 = sift.detectAndCompute(edges2, None)

# # 5. Match descriptors
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1, des2, k=2)

# # 6. Apply ratio test (Lowe's ratio test)
# good_matches = []
# for m, n in matches:
#     if m.distance < 0.6 * n.distance:
#         good_matches.append(m)

# # 7. Draw matches
# # img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# # img_matches = cv2.drawMatches(matching_lane_image, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# # img_matches = cv2.drawMatches(gray1, kp1, gray2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# img_matches = cv2.drawMatches(edges1, kp1, edges2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


# # 8. Display or save the result
# display_image(img_matches)