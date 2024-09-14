import numpy as np

def evaluate_results_task1(predictions_path,ground_truth_path,verbose = 0):
    total_correct_number_pins = 0
    
    #for i in range(1,26):
    for i in range(1,4):
        correct_number_pins = 0
    
        try:
            if(i<10):
                name = '0' + str(i)
            else:
                name = str(i)
    
            filename_predictions = predictions_path  + name + "_predicted.txt"
            filename_ground_truth = ground_truth_path + name + "_gt.txt"

            #print(filename_predictions,filename_ground_truth)
         
            p = open(filename_predictions,"rt")
            #print("p = ", p)
            gt = open(filename_ground_truth,"rt")
            #print("gt = ", gt)
        
            correct_number_pins = 1

            #read the first line - number of queries
            p_line = p.readline()
            gt_line = gt.readline()
    


            nb_gt_lines = int(gt_line[:1])
            #read the next nb_gt_lines
            for q in range(1,nb_gt_lines+1):
                p_line = p.readline()
                
                gt_line = gt.readline()
                #print(p_line,gt_line)

                if (p_line != gt_line):
                    correct_prediction = 0
                    break

            p.close()
            gt.close()        

        except:
            print("Error")


        if verbose:
            print("Task 1 - Classifying positions standing pins: for test example number ", str(i), " the prediction is :", (1-correct_number_pins) * "in" + "correct \n")
               
        total_correct_number_pins = total_correct_number_pins + correct_number_pins
        points = total_correct_number_pins * 0.09
        
    return total_correct_number_pins, points

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def compute_percentage_tracking(gt_bboxes, predicted_bboxes, num_frames,verbose=0):
    """
    This function compute the percentage of detected bounding boxes based on the ground-truth bboxes and the predicted ones.
    :param gt_bboxes. The ground-truth bboxes with the format: frame_idx, x_min, y_min, x_max, y_max.
    :param predicted_bboxes. The predicted bboxes with the format: frame_idx, x_min, y_min, x_max, y_max
    :param num_frames. The total number of frames in the video.
    """
    
    num_frames = int(num_frames)
    
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    gt_dict = {}
    for gt_box in gt_bboxes:
        gt_dict[gt_box[0]] = gt_box[1:]
    
    pred_dict = {}
    for pred_bbox in predicted_bboxes:
        pred_dict[pred_bbox[0]] = pred_bbox[1:]
        
    for i in range(num_frames):
        if gt_dict.get(i, None) is None and pred_dict.get(i, None) is None: # the bowling bowl is not on the lane
            tn += 1 
        
        elif gt_dict.get(i, None) is not None and pred_dict.get(i, None) is None: # the bowling ball is not detected
            fn += 1
            
        elif gt_dict.get(i, None) is None and pred_dict.get(i, None) is not None: # the bowling ball is not on the lane, but it is 'detected'
            fp += 1
            
        elif gt_dict.get(i, None) is not None and pred_dict.get(i, None) is not None: # the bowling ball is on the lane and it is detected
            
            iou = bb_intersection_over_union(gt_dict[i], pred_dict[i])
            if iou >= 0.3:
                tp += 1
            else:
                fp += 1 
                         
    if verbose:
        print(f'tp = {tp}, tn = {tn}, fp = {fp},fn = {fn}')
    assert tn + fn + tp + fp == num_frames
    perc = (tp + tn) / (tp + fp + tn + fn)

    return perc

def evaluate_results_task2(predictions_path,ground_truth_path, verbose = 0):
    total_correct_tracked_videos = 0
    for i in range(1,16):
    # for i in range(1,2):

        correct_tracked_video = 0
        
        try:

            if(i<10):
                name = '0' + str(i)
            else:
                name = str(i)
            
            filename_predictions = predictions_path + "/" + name + "_predicted.txt"
            filename_ground_truth = ground_truth_path + "/" + name + "_gt.txt"

            p = np.loadtxt(filename_predictions)            
            predicted_bboxes = p[1:]
            gt = np.loadtxt(filename_ground_truth)
            gt_bboxes = gt[1:]
            num_frames = gt[0][0]
            percentage = compute_percentage_tracking(gt_bboxes, predicted_bboxes, num_frames,verbose)

            correct_tracked_video = 1
            if percentage < 0.8:
                correct_tracked_video = 0
        
            
            if verbose:
                print("percentage = ", percentage)
                print("Task 2 - Tracking the bowling ball: for test example number ", str(i), " the prediction is :", (1-correct_tracked_video) * "in" + "correct", "\n")
        
            total_correct_tracked_videos = total_correct_tracked_videos + correct_tracked_video
        
        except:
            print("Error")

        points = total_correct_tracked_videos * 0.15
        
    return total_correct_tracked_videos,points 



def evaluate_results_task3(predictions_path,ground_truth_path,verbose = 0):
    total_correct_number_pins = 0    

    for i in range(1,16):
    # for i in range(1,2):
        correct_number_pins = 1     
        try:
            if(i<10):
                name = '0' + str(i)
            else:
                name = str(i)
    
            filename_predictions = predictions_path + name + "_predicted.txt"
            filename_ground_truth = ground_truth_path + name + ".txt"
         
            p = open(filename_predictions,"rt")
            gt = open(filename_ground_truth,"rt")               

            #read the first line - number of standing pins before
            p_pred = int(p.readline())
            gt_pred = int(gt.readline())

            if (p_pred != gt_pred):
                correct_number_pins = 0

            #read the second line - number of standing pins after
            p_pred = int(p.readline())
            gt_pred = int(gt.readline())

            if (p_pred != gt_pred):
                correct_number_pins = 0
        
            p.close()
            gt.close()        

        except:
            print("Error")


        if verbose:
            print("Task 3 - Counting standing pins before/after in video for test example number ", str(i), " the prediction is :", (1-correct_number_pins) * "in" + "correct for number of pins" + "\n")
               
        total_correct_number_pins = total_correct_number_pins + correct_number_pins     
        points = total_correct_number_pins * 0.05

    return total_correct_number_pins, points


#change this on your machine
predictions_path_root = "F:/Master/An1/sem2/cv/Project-Computer-Vision-Bowling/Mihai_Popa_407/"
ground_truth_path_root = "F:/Master/An1/sem2/cv/Project-Computer-Vision-Bowling/data/test/ground-truth/"

#task1
verbose = 0
predictions_path = predictions_path_root + "Task1/"
ground_truth_path = ground_truth_path_root + "Task1/"
# total_correct_number_pins, points_task1 = evaluate_results_task1(predictions_path,ground_truth_path,verbose)

# print("Task 1 = ", points_task1)



#task2
verbose = 0
predictions_path = predictions_path_root + "Task2/"
ground_truth_path = ground_truth_path_root + "Task2/"
total_correct_tracked_videos_task2,points_task2 = evaluate_results_task2(predictions_path,ground_truth_path,verbose)
print("Task 2 = ", points_task2)

#task3
verbose = 0
predictions_path = predictions_path_root + "Task3/"
ground_truth_path = ground_truth_path_root + "Task3/"
total_correct_number_pins, points_task3 = evaluate_results_task3(predictions_path,ground_truth_path,verbose)
print("Task 3 = ", points_task3)

#print("Task 1 = ", points_task1, "\nTask 2 = ",points_task2, "\nTask 3 = ", points_task3, "\nTo to add 0.5 points ex officio")