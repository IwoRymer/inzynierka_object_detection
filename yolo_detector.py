#model = YOLO("yolov8n.pt", device='gpu')

import cv2
import ultralytics
from ultralytics import YOLO, ASSETS
ultralytics.checks()
#import time
import numpy as np
import os
import time

import torch
import radar_functions
import label
from copy import deepcopy

files_named = 'detect_human_end_of_day'
files_named_alt = 'detect_human_end_of_day'
dir_path = 'human_day_CR/'

#files_named = 'car_detection_night_' 
#files_named_alt = 'car_detection_night_' 
#dir_path = 'car_night_day_3/'



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

model_core = YOLO("yolo11m.pt").to(device)
model_child = YOLO("yolo11s.pt").to(device)
model_test = YOLO("yolo11x.pt").to(device)
mode_above = 0
draw_x_above = False
elapsed_time_above = 0
limit_above = 1.5 # 1.5 car, 0.5 human
label_mode_above = False
bias_above =80
labels_exist_above = 1
treshold_above = 0.3
#model = YOLO("yolo11n.pt").to('cpu')

#-----------------------
camera_x = 1920 # width of camera picture
camera_z = 1080 # height of camera picture

cameraFOV = 77
cameraFOV_rad = np.deg2rad(cameraFOV)

camera_center_z = int(camera_z/2)
camera_center_x = int(camera_x/2)

P = camera_x/2 # half of the screen where detected object can 
FOV_half_rad =  cameraFOV_rad/2
tan_FOV_half = np.tan(FOV_half_rad)
false_distance = (camera_x/2)/tan_FOV_half
#-------------------



def safe_txt(string, combined_array):

    # Save to a text file
    np.savetxt(string, combined_array, fmt='%f', delimiter=',', header='Array1, Array2', comments='')

    print("Arrays saved to txt")

def read_txt(path):
    loaded_array = np.loadtxt(path, delimiter=',', skiprows=1)
    if loaded_array is None:
        return None
    
    if loaded_array.ndim == 1:
        array1 = np.array([loaded_array[0]])
        array1 = np.array([loaded_array[1]])
    else:
        
    # Separate the columns if you want to access each array individually
        array1 = loaded_array[:, 0]  # First column
        array2 = loaded_array[:, 1]  # Second column

        return array1, array2


def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf, verbose=False, show_labels=False)
    else:
        results = chosen_model.predict(img, conf=conf, verbose=False, show_labels=False)

    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5):
    #print(len(img))
    #print(len(img[0]))
    results = predict(chosen_model, img, classes, conf=conf)
    i = 0
    boxes = []
    detected_classes = [] # 0 - human, 2 - car
    confidences = []
    for result in results:
        for box in result.boxes:
            #cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
             #           (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            #cv2.putText(img, f"{result.names[int(box.cls[0])]}",
             #           (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
              #          cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), text_thickness)

            #if i == 0:
                #print(result.boxes)
             #   pass
            #cv2.putText(img, f"{result.boxes.conf[i]:.2f}",
             #       (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 20),
              #      cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), text_thickness)
            
            bb_this = [int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])]  # corner x1, corner y1 ,  corner x2, corner y2
            detected_class = int(box.cls[0])
            confidence = float(result.boxes.conf[i])
            
            boxes.append(bb_this)
            detected_classes.append(detected_class)
            confidences.append(confidence)
            
            
            i+=1
    return results, boxes, detected_classes, confidences


def draw_detect_info(img, boxes, detected_classes, confidences, colour = 'blue', rectangle_thickness=2, text_thickness=2):
    
    if colour == 'blue':
        color = (255, 0, 0)
    elif colour == 'red':
        color = (0, 0, 255)
    elif colour == 'white':
        color = (255, 255, 255)
    else:
        color = (0, 255, 255)
    
    for i in range(0, len(boxes)):
        
        if detected_classes[i] == 0:
            obiekt = 'person'
        elif detected_classes[i] == 2:
            obiekt = 'car'
            
        #cv2.putText(img, f"{result[0].names[detected_classes[i]]}", (boxes[i][0], boxes[i][1] - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), text_thickness)
        cv2.putText(img, f"{obiekt}", (boxes[i][0], boxes[i][1] - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), text_thickness)
        cv2.rectangle(img, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), color, rectangle_thickness)
        cv2.putText(img, f"{confidences[i]:.2f}", (boxes[i][0], boxes[i][1] - 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), text_thickness)
    return img

def draw_detections(pixels, image, bias=0):
    camera_center_z_alt = camera_center_z - bias
    for i in pixels:
        first = 0 # blue
        second = 0 #green
        third = 255 # red

        # centrum
        image[camera_center_z_alt][camera_center_x + i][0] = first
        image[camera_center_z_alt][camera_center_x + i][1] = second
        image[camera_center_z_alt][camera_center_x + i][2] = third
        
        image[camera_center_z_alt][camera_center_x + i - 3][0] = first
        image[camera_center_z_alt][camera_center_x + i - 3][1] = second
        image[camera_center_z_alt][camera_center_x + i - 3][2] = third
        
        image[camera_center_z_alt][camera_center_x + i + 3][0] = first
        image[camera_center_z_alt][camera_center_x + i + 3][1] = second
        image[camera_center_z_alt][camera_center_x + i + 3][2] = third
        
        image[camera_center_z_alt+1][camera_center_x + i - 3][0] = first
        image[camera_center_z_alt+1][camera_center_x + i - 3][1] = second
        image[camera_center_z_alt+1][camera_center_x + i - 3][2] = third
        
        image[camera_center_z_alt+1][camera_center_x + i + 3][0] = first
        image[camera_center_z_alt+1][camera_center_x + i + 3][1] = second
        image[camera_center_z_alt+1][camera_center_x + i + 3][2] = third
        
        image[camera_center_z_alt-1][camera_center_x + i - 3][0] = first
        image[camera_center_z_alt-1][camera_center_x + i - 3][1] = second
        image[camera_center_z_alt-1][camera_center_x + i - 3][2] = third
        
        image[camera_center_z_alt-1][camera_center_x + i + 3][0] = first
        image[camera_center_z_alt-1][camera_center_x + i + 3][1] = second
        image[camera_center_z_alt-1][camera_center_x + i + 3][2] = third
        
        image[camera_center_z_alt+2][camera_center_x + i - 3][0] = first
        image[camera_center_z_alt+2][camera_center_x + i - 3][1] = second
        image[camera_center_z_alt+2][camera_center_x + i - 3][2] = third
        
        image[camera_center_z_alt+2][camera_center_x + i + 3][0] = first
        image[camera_center_z_alt+2][camera_center_x + i + 3][1] = second
        image[camera_center_z_alt+2][camera_center_x + i + 3][2] = third
        
        image[camera_center_z_alt-2][camera_center_x + i - 3][0] = first
        image[camera_center_z_alt-2][camera_center_x + i - 3][1] = second
        image[camera_center_z_alt-2][camera_center_x + i - 3][2] = third
        
        image[camera_center_z_alt-2][camera_center_x + i + 3][0] = first
        image[camera_center_z_alt-2][camera_center_x + i + 3][1] = second
        image[camera_center_z_alt-2][camera_center_x + i + 3][2] = third
        
        image[camera_center_z_alt+3][camera_center_x + i - 3][0] = first
        image[camera_center_z_alt+3][camera_center_x + i - 3][1] = second
        image[camera_center_z_alt+3][camera_center_x + i - 3][2] = third
        
        image[camera_center_z_alt+3][camera_center_x + i + 3][0] = first
        image[camera_center_z_alt+3][camera_center_x + i + 3][1] = second
        image[camera_center_z_alt+3][camera_center_x + i + 3][2] = third
        
        image[camera_center_z_alt-3][camera_center_x + i - 3][0] = first
        image[camera_center_z_alt-3][camera_center_x + i - 3][1] = second
        image[camera_center_z_alt-3][camera_center_x + i - 3][2] = third
        
        image[camera_center_z_alt-3][camera_center_x + i + 3][0] = first
        image[camera_center_z_alt-3][camera_center_x + i + 3][1] = second
        image[camera_center_z_alt-3][camera_center_x + i + 3][2] = third
        
        image[camera_center_z_alt-3][camera_center_x + i + 0][0] = first
        image[camera_center_z_alt-3][camera_center_x + i + 0][1] = second
        image[camera_center_z_alt-3][camera_center_x + i + 0][2] = third
        
        image[camera_center_z_alt+3][camera_center_x + i + 0][0] = first
        image[camera_center_z_alt+3][camera_center_x + i + 0][1] = second
        image[camera_center_z_alt+3][camera_center_x + i + 0][2] = third
        
        image[camera_center_z_alt-3][camera_center_x + i + 1][0] = first
        image[camera_center_z_alt-3][camera_center_x + i + 1][1] = second
        image[camera_center_z_alt-3][camera_center_x + i + 1][2] = third
        
        image[camera_center_z_alt+3][camera_center_x + i + 1][0] = first
        image[camera_center_z_alt+3][camera_center_x + i + 1][1] = second
        image[camera_center_z_alt+3][camera_center_x + i + 1][2] = third
        
        image[camera_center_z_alt-3][camera_center_x + i - 1][0] = first
        image[camera_center_z_alt-3][camera_center_x + i - 1][1] = second
        image[camera_center_z_alt-3][camera_center_x + i - 1][2] = third
        
        image[camera_center_z_alt+3][camera_center_x + i - 1][0] = first
        image[camera_center_z_alt+3][camera_center_x + i - 1][1] = second
        image[camera_center_z_alt+3][camera_center_x + i - 1][2] = third
        
        image[camera_center_z_alt-3][camera_center_x + i + 2][0] = first
        image[camera_center_z_alt-3][camera_center_x + i + 2][1] = second
        image[camera_center_z_alt-3][camera_center_x + i + 2][2] = third
        
        image[camera_center_z_alt+3][camera_center_x + i + 2][0] = first
        image[camera_center_z_alt+3][camera_center_x + i + 2][1] = second
        image[camera_center_z_alt+3][camera_center_x + i + 2][2] = third
        
        image[camera_center_z_alt-3][camera_center_x + i - 2][0] = first
        image[camera_center_z_alt-3][camera_center_x + i - 2][1] = second
        image[camera_center_z_alt-3][camera_center_x + i - 2][2] = third
        
        image[camera_center_z_alt+3][camera_center_x + i - 2][0] = first
        image[camera_center_z_alt+3][camera_center_x + i - 2][1] = second
        image[camera_center_z_alt+3][camera_center_x + i - 2][2] = third
        
        
    return image

def draw_detections_bb(pixels, image, h, w, bias=0):
    img_with_box = image
    camera_center_z_alt = camera_center_z - bias
    for i in pixels:
        y = camera_center_z_alt - int(h/2)
        x = i - int(w/2) + camera_center_x
        if (i + camera_center_x) > int(w/2) & (i + camera_center_x) < (camera_x - int(w/2)): #jezeli nie ucina
            img_with_box = cv2.rectangle(img_with_box, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        elif i < 0:
            #lewa strona
            img_with_box = cv2.rectangle(img_with_box, (0, y), (0 + w, y + h), (0, 0, 255), 2)
        else:
            #prawa strona
            img_with_box = cv2.rectangle(img_with_box, (camera_x-1-w, y), (camera_x-1, y + h), (0, 0, 255), 2)
            
        

    return img_with_box


def bb_percentage_of_area(boxA, boxB): #boxA - radar 
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
	#boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	area = interArea / float(boxAArea)
	# return the intersection over union value
	return area

def activate_yolo(image, detection_treshold, pixels_detected = None, draw=False, time_del=0, draw_detected_pts = False):
    
    start_time = time.time()
    result, boxes, detected_classes, confidences = predict_and_detect(model_core, image, classes=[0, 2], conf=detection_treshold)
    end_time = time.time()

    
    elapsed_time = end_time - start_time
    draw_time = None
    
    if draw:

        result_img = draw_detect_info(image, boxes, detected_classes, confidences, result)

        if pixels_detected is not None and draw_detected_pts:
            result_img = draw_detections(pixels_detected, result_img, bias)
            result_img = draw_detections_bb(pixels_detected, result_img, 640, 640, bias)
        
        
        if camera_x == 1920:
            # Define the desired dimensions for the resized image
            width = 1280
            height = 720

            # Resize the image
            resized_image = cv2.resize(result_img, (width, height))
            image_displayed = "image num {}".format(real_num)
            cv2.putText(resized_image, image_displayed, (800, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)
            cv2.imshow("image", resized_image)
            cv2.waitKey(time_del)
        elif camera_x == 1920 and label_mode:
            resized_image = cv2.resize(result_img, (camera_x/2, camera_z/2))
            image_displayed = "image num {}".format(real_num)
            cv2.putText(resized_image, image_displayed, (800, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)
            cv2.imshow("image", resized_image)
            cv2.waitKey(time_del)
        
        else:
            image_displayed = "image num {}".format(real_num)
            cv2.putText(result_img, image_displayed, (800, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)
            cv2.imshow("image", result_img)
            cv2.waitKey(time_del)
        draw_time_b = time.time()
        draw_time = draw_time_b - end_time
    return elapsed_time, draw_time



def yolo_single_acivation(model, detection_treshold, image, classes=[0, 2]):
    results, boxes, detected_classes, confidences = predict_and_detect(model, image, classes, detection_treshold)
    return results, boxes, detected_classes, confidences


def yolo_core_and_children(model_core, model_child, image, detection_treshold, radar_pixels=None, width=640, height=640, bias = 0, draw=False, time_del=0, draw_detected_pts = False, classes=[0,2]):
    
    start = time.time()
    _, boxes_r, detected_classes_r, confidences_r = predict_and_detect(model_core, image, classes, detection_treshold)
        
    boxes_fusion_list = deepcopy(boxes_r)
    detected_classes_fusion_list = deepcopy(detected_classes_r)
    confidences_fusion_list = deepcopy(confidences_r)
    
    boxes_children_list = []
    detected_classes_children_list = []
    confidences_children_list = []
     
    
    if radar_pixels is not None and len(radar_pixels) > 0:
        top = int(camera_center_z - height/2 - bias)
        bottom = top + height
        #print('top', top)
        #print('bottom', bottom)
        
        if len(detected_classes_r) > 0:
            core = True
        else:
            core = False
        
        for i in radar_pixels:
            
            pozycja = i + camera_center_x
            #print('pozycja =', pozycja)
            
            if pozycja < width/2:
                l = 0
                sub_img = image[top:bottom, :width]
                #print(len(sub_img), 'if')
            elif pozycja > (camera_x - width/2):
                l = camera_x - width
                #print('l =', l)
                sub_img = image[top:bottom, l:]
                #print(len(sub_img), 'elif')
            else:
                l = int(pozycja-width/2)
                #print('l =', l)
                r = int(pozycja+width/2)
                #print('r = ', r)
                sub_img = image[top:bottom, l:r]
                #print(len(sub_img), 'else')
            results_d, boxes_d, detected_classes_d, confidences_d = predict_and_detect(model_child, sub_img, classes, detection_treshold)
            
            if len(detected_classes_d) > 0:   # przesuniecie
                
                for j in range(0, len(detected_classes_d)):
                    
                    boxes_d[j][0] = boxes_d[j][0] + l
                    boxes_d[j][1] = boxes_d[j][1] + top
                    boxes_d[j][2] = boxes_d[j][2] + l
                    boxes_d[j][3] = boxes_d[j][3] + top
            
            # porownaj wykrycia
            
            if len(detected_classes_d) > 0 and core:
                already_detected = [False for i in range(0, len(detected_classes_d))]
                
                for k in range(0, len(detected_classes_d)):
                    best_ratio = 0
                    
                    for rdzen in range(0, len(detected_classes_r)):
                        
                        if detected_classes_r[rdzen] == detected_classes_d[k]:
                            ratio = bb_percentage_of_area(boxes_d[k], boxes_r[rdzen])
                            if ratio > best_ratio and ratio > 0.75:
                                best_ratio = ratio
                                already_detected[k] = True
                                
                                #popraw conf
                                confidences_fusion_list[rdzen] = max(confidences_fusion_list[rdzen], confidences_d[k])
                    
                    if k > 0:
                        # wiele pkt wykrytych przez dziecko
                        for dziecko in range(0, k):
                            if detected_classes_d[dziecko] == detected_classes_d[k]:
                                ratio = bb_percentage_of_area(boxes_d[k], boxes_d[dziecko])
                                if ratio > best_ratio and ratio > 0.75:
                                    best_ratio = ratio
                                    already_detected[k] = True
                                    #popraw conf
                                    confidences_d[dziecko] = max(confidences_d[dziecko], confidences_d[k])
                    if not already_detected[k]:
                        boxes_fusion_list.append(boxes_d[k])
                        detected_classes_fusion_list.append(detected_classes_d[k])
                        confidences_fusion_list.append(confidences_d[k])
                        
                        boxes_children_list.append(boxes_d[k])
                        detected_classes_children_list.append(detected_classes_d[k])
                        confidences_children_list.append(confidences_d[k])
    
    dane_rdzen = [boxes_r, detected_classes_r, confidences_r]
    dane_fuzja = [boxes_fusion_list, detected_classes_fusion_list, confidences_fusion_list]
    dane_dzieci = [boxes_children_list, detected_classes_children_list, confidences_children_list]
    
    draw_time_a = time.time()
    
    
    end = time.time()
    
    if draw:

        result_img = draw_detect_info(image, boxes_r, detected_classes_r, confidences_fusion_list, 'blue')
        result_img = draw_detect_info(result_img, boxes_children_list, detected_classes_children_list, confidences_children_list, 'yellow')

        if radar_pixels is not None and draw_detected_pts:
            result_img = draw_detections(radar_pixels, result_img, bias)
            result_img = draw_detections_bb(radar_pixels, result_img, 640, 640, bias)
        
        
        if camera_x == 1920:
            # Define the desired dimensions for the resized image
            width = 1280
            height = 720

            # Resize the image
            resized_image = cv2.resize(result_img, (width, height))
            image_displayed = "image num {}".format(real_num)
            cv2.putText(resized_image, image_displayed, (800, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)
            cv2.imshow("image", resized_image)
            cv2.waitKey(time_del)
        
        else:
            image_displayed = "image num {}".format(real_num)
            cv2.putText(result_img, image_displayed, (800, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)
            cv2.imshow("image", result_img)
            cv2.waitKey(time_del)
    draw_time_b = time.time()
    draw_time = draw_time_b - draw_time_a
    action_time = end - start
    
    return action_time, draw_time, dane_rdzen, dane_fuzja, dane_dzieci



def calculate_values(conf_x, class_x, conf_core, class_core, conf_fusion, class_fusion, cars, people, movement, movement_real, conf_people, conf_cars):
    
    # conf people
    # conf cars - number of consensus detections
    # 0 - person, 2 - car
    
    if movement > 0:
        movement_real += 1
    
    detected_x_people = 0
    detected_x_cars = 0
    x_car_conf = []
    x_people_conf = []
    
    detected_core_people = 0
    detected_core_cars = 0
    core_car_conf = []
    core_people_conf = []
    
    detected_fusion_people = 0
    detected_fusion_cars = 0
    fusion_car_conf = []
    fusion_people_conf = []

    for i in range(0, len(conf_x)):
        if class_x[i] == 0:
            detected_x_people += 1
            x_people_conf.append(conf_x[i])
        elif class_x[i] == 2:
            detected_x_cars += 1
            x_car_conf.append(conf_x[i])
    
    for i in range(0, len(conf_core)):
        if class_core[i] == 0:
            detected_core_people += 1
            core_people_conf.append(conf_core[i])
        elif class_core[i] == 2:
            detected_core_cars += 1
            core_car_conf.append(conf_core[i])

    for i in range(0, len(conf_fusion)):
        if class_fusion[i] == 0:
            detected_fusion_people += 1
            fusion_people_conf.append(conf_fusion[i])
            
        elif class_fusion[i] == 2:
            detected_fusion_cars += 1
            fusion_car_conf.append(conf_fusion[i])

    
    detected_x_people = min(detected_x_people, people)
    detected_core_people = min(detected_core_people, people)
    detected_fusion_people = min(detected_fusion_people, people)
    #print("x = ", detected_x_people)
    #print("core = ", detected_core_people)
    #print("fusion = ", detected_fusion_people)
    #print('real = ', people)
    
    detected_x_cars = min(detected_x_cars, cars)
    detected_core_cars = min(detected_core_cars, cars)
    detected_fusion_cars = min(detected_fusion_cars, cars)
    
    confidence_metric_x_people = 0
    confidence_metric_x_cars = 0
    x_car_conf.sort(reverse=True)
    x_people_conf.sort(reverse=True)
    
    
    confidence_metric_core_people = 0
    confidence_metric_core_cars = 0
    core_car_conf.sort(reverse=True)
    core_people_conf.sort(reverse=True)
    
    
    confidence_metric_fusion_people = 0
    confidence_metric_fusion_cars = 0
    fusion_car_conf.sort(reverse=True)
    fusion_people_conf.sort(reverse=True)
    

    people_min = min(people, detected_x_people, detected_core_people, detected_fusion_people)
    car_min = min(cars, detected_x_cars, detected_core_cars, detected_fusion_cars)
    
    conf_people += people_min
    conf_cars += car_min
    
    
    for j in range(0, people_min):
        confidence_metric_x_people += x_people_conf[j]
        confidence_metric_core_people += core_people_conf[j]
        confidence_metric_fusion_people += fusion_people_conf[j]
        
        #print('x = ', confidence_metric_x_people)
        #print("core = ", confidence_metric_core_people)
        #print("fusion = ", confidence_metric_fusion_people)
        
    for j in range(0, car_min):
        confidence_metric_x_cars += x_car_conf[j]
        confidence_metric_core_cars += core_car_conf[j]
        confidence_metric_fusion_cars += fusion_car_conf[j]
    
    Confidence_metric_fusion = [confidence_metric_fusion_people, confidence_metric_fusion_cars]
    Confidence_metric_x = [confidence_metric_x_people, confidence_metric_x_cars]
    Confidence_metric_core = [confidence_metric_core_people, confidence_metric_core_cars]
    
    Detected_x = [detected_x_people, detected_x_cars]
    Detected_core = [detected_core_people, detected_core_cars]
    Detected_fusion = [detected_fusion_people, detected_fusion_cars]

    
    return movement_real, Confidence_metric_x, Confidence_metric_core, Confidence_metric_fusion, Detected_x, Detected_core, Detected_fusion, conf_people, conf_cars



# ----- modifiable parameters

elapsed_time = elapsed_time_above
mode = mode_above # 0 = image, 1 = sub image (640), other = sub img (960)
draw = True                  # activate drawing
draw_pts = True   # draw radar points
draw_x = draw_x_above # draw_detections_from_x _on seperate  # WAIT KEY 0!
detection_treshold = treshold_above  # confidence!
limit = limit_above  # 0.5 - human, 1.5 car?

eliminate_near_points = True    # combine multipoints into one
filter_pixels = True    # filter detection launches on the same azymuth
filter_pixels_lim = 5
label_mode = label_mode_above   # generate labels
labels_exist = labels_exist_above  # 1 if label files in folder
#x_mask = [2.05, -1.72, -4.50]        # kolejne punkty w masce
#y_mask = [10.84, 10.90, 10.17]
x_mask = [-4.87, 4.17]
y_mask = [10.00, 10.03]
activate_masks = False

repeat_detections = True
bias = bias_above

readout_load_txt = 0
readout_load_img = 0
num_run = 1
# ------ mod parameters

time1 = time.time()
for k in range(0, num_run):
    file_count = 0
    total_time = 0
    point_count = 0
    point_count_red = 0
    point_count_red2 = 0
    max_pts1 = 0
    max_pts2 = 0
    max_pts3 = 0
    real_num = 0
    detected_motion = 0
    total_time_loop = 0
    total_draw_time = 0
    movement_real = 0
    conf_people = 0
    conf_cars = 0
    
    prev_pixels_radar = None
    pixels_radar2 = None
    last_frame_was_movement = False
    this_frame_is_movement = False
    
    # ------------------
    x_version_conf_people_count = 0
    core_version_conf_people_count = 0
    fusion_version_conf_people_count = 0
    
    x_version_conf_cars_count = 0
    core_version_conf_cars_count = 0
    fusion_version_conf_cars_count = 0
    
    detected_x_people_count = 0
    detected_core_people_count = 0
    detected_fusion_people_count = 0
    
    detected_x_cars_count = 0
    detected_core_cars_count = 0
    detected_fusion_cars_count = 0
    
    people_count = 0
    car_count = 0
    
    child_check = 0
    check_counter = 0
    
    # ----------------

    
    for path in os.scandir(dir_path):
        
        
        if path.is_file():
            file_count += 1
        if file_count % (2 + labels_exist) == 0:
            
            
            time_load_txt_1 = time.time()
            file_path = dir_path + files_named_alt + str(real_num) + '.txt'
            arrays = read_txt(file_path)
            time_load_txt_2 = time.time()
            readout_load_txt += time_load_txt_2 - time_load_txt_1
        
            if arrays is not None:
                this_frame_is_movement = True
                
                x, y = arrays
                point_count += len(x)
                max_pts1 = max(max_pts1, len(x))
                if activate_masks:
                    x, y = radar_functions.filter_clutter_from_mask(x_mask, y_mask, x, y)
                if eliminate_near_points:
                    new_x, new_y = radar_functions.limit_points(x, y, limit)
                else:
                    new_x, new_y = x, y
                point_count_red += len(new_x)
                max_pts2 = max(max_pts2, len(new_x))
            
                pixels_radar = radar_functions.detection_2_pixels(new_x, new_y, false_distance, camera_center_x)
                if filter_pixels:
                    pixels_radar2 = radar_functions.filter_pixels_after(pixels_radar, filter_pixels_lim)
                else:
                    pixels_radar2 = pixels_radar
                point_count_red2 += len(pixels_radar2)
                max_pts3 = max(max_pts3, len(pixels_radar2))
            
            else:
                x = None
                this_frame_is_movement = False
        
            if x is None: # nie wykryto nic przez radar
                time_load_img_1 = time.time()
                file_path_img = dir_path + files_named + str(real_num) + '.jpg'
                image = cv2.imread(file_path_img)
                time_load_img_2 = time.time()
                readout_load_img += time_load_img_2 - time_load_img_1
                if last_frame_was_movement and repeat_detections:
                    check_counter += 1
                    pixels_radar2 = prev_pixels_radar
                    prev_pixels_radar = None
        
                if mode == 0:
                    width = 640
                    height = 640
                    image_copy = deepcopy(image)
                    elapsed, draw_time, rdzen, fuzja, dzieci = yolo_core_and_children(model_core, model_child, image, detection_treshold, pixels_radar2, width, height, bias, draw, elapsed_time, draw_pts, [0, 2])
                    _, boxes_x, detected_classes_x, confidences_x = yolo_single_acivation(model_test, detection_treshold, image_copy, [0, 2])
                    if draw_x:
                        image_copy = draw_detect_info(image_copy, boxes_x, detected_classes_x, confidences_x, 'white', 1, 1)
                        width = 1280
                        height = 720

                        # Resize the image
                        resized_image = cv2.resize(image_copy, (width, height))
                        image_displayed = "image num {}".format(real_num)
                        cv2.putText(resized_image, image_displayed, (800, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)
                        cv2.imshow("image_x", resized_image)
                        cv2.waitKey(elapsed_time)
                    
                    child_check += len(dzieci[2])
                    if label_mode:
                        cars, people, movement = label.label_photo_return(real_num)
                    else:
                        label_filename = dir_path + f"label_{real_num}.txt"
                        cars, people, movement = label.read_label_return(label_filename)
                    people_count += people
                    car_count += cars
                    movement_real, Confidence_metric_x, Confidence_metric_core, Confidence_metric_fusion, Detected_x, Detected_core, Detected_fusion, conf_people, conf_cars = calculate_values(confidences_x, detected_classes_x,
                                                                                                                                                                                                rdzen[2], rdzen[1], fuzja[2], fuzja[1],
                                                                                                                                                                                                cars, people, movement, movement_real,
                                                                                                                                                                                                conf_people, conf_cars)
                    x_version_conf_people, x_version_conf_cars = Confidence_metric_x
                    core_version_conf_people, core_version_conf_cars = Confidence_metric_core
                    fusion_version_conf_people, fusion_version_conf_cars = Confidence_metric_fusion
                    
                    detected_x_people, detected_x_cars = Detected_x
                    detected_core_people, detected_core_cars = Detected_core
                    detected_fusion_people, detected_fusion_cars = Detected_fusion
                    
                    x_version_conf_people_count += x_version_conf_people
                    x_version_conf_cars_count += x_version_conf_cars
                    core_version_conf_people_count += core_version_conf_people
                    core_version_conf_cars_count += core_version_conf_cars
                    fusion_version_conf_people_count += fusion_version_conf_people
                    fusion_version_conf_cars_count += fusion_version_conf_cars
                    
                    detected_x_people_count += detected_x_people
                    detected_x_cars_count += detected_x_cars
                    detected_core_people_count += detected_core_people
                    detected_core_cars_count += detected_core_cars
                    detected_fusion_people_count += detected_fusion_people
                    detected_fusion_cars_count += detected_fusion_cars
                    
                    

                    # note: class 0 = person, 2 = car
                    #elapsed, draw_time = activate_yolo(image, detection_treshold, None, draw, elapsed_time, False)
                elif mode == 1:   # depricated
                    bias = min(60, bias)
                    bias = max(-60, bias)
                    width = 960
                    height = 960
                    image_copy = deepcopy(image)
                    elapsed, draw_time, rdzen, fuzja, dzieci = yolo_core_and_children(model_core, model_child, image, detection_treshold, pixels_radar2, width, height, bias, draw, elapsed_time, draw_pts, [0, 2])
                    _, boxes_x, detected_classes_x, confidences_x = yolo_single_acivation(model_test, detection_treshold, image_copy, [0, 2])
                    if draw_x:
                        image_copy = draw_detect_info(image_copy, boxes_x, detected_classes_x, confidences_x, 'white', 1, 1)
                        width = 1280
                        height = 720

                        # Resize the image
                        resized_image = cv2.resize(image_copy, (width, height))
                        image_displayed = "image num {}".format(real_num)
                        cv2.putText(resized_image, image_displayed, (800, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)
                        cv2.imshow("image_x", resized_image)
                        cv2.waitKey(elapsed_time)
                    child_check += len(dzieci[2])
                    if label_mode:
                        cars, people, movement = label.label_photo_return(real_num)
                    else:
                        label_filename = dir_path + f"label_{real_num}.txt"
                        cars, people, movement = label.read_label_return(label_filename)
                    people_count += people
                    car_count += cars
                    movement_real, Confidence_metric_x, Confidence_metric_core, Confidence_metric_fusion, Detected_x, Detected_core, Detected_fusion, conf_people, conf_cars = calculate_values(confidences_x, detected_classes_x,
                                                                                                                                                                                                rdzen[2], rdzen[1], fuzja[2], fuzja[1],
                                                                                                                                                                                                cars, people, movement, movement_real,
                                                                                                                                                                                                conf_people, conf_cars)
                    x_version_conf_people, x_version_conf_cars = Confidence_metric_x
                    core_version_conf_people, core_version_conf_cars = Confidence_metric_core
                    fusion_version_conf_people, fusion_version_conf_cars = Confidence_metric_fusion
                    
                    detected_x_people, detected_x_cars = Detected_x
                    detected_core_people, detected_core_cars = Detected_core
                    detected_fusion_people, detected_fusion_cars = Detected_fusion
                    
                    x_version_conf_people_count += x_version_conf_people
                    x_version_conf_cars_count += x_version_conf_cars
                    core_version_conf_people_count += core_version_conf_people
                    core_version_conf_cars_count += core_version_conf_cars
                    fusion_version_conf_people_count += fusion_version_conf_people
                    fusion_version_conf_cars_count += fusion_version_conf_cars
                    
                    detected_x_people_count += detected_x_people
                    detected_x_cars_count += detected_x_cars
                    detected_core_people_count += detected_core_people
                    detected_core_cars_count += detected_core_cars
                    detected_fusion_people_count += detected_fusion_people
                    detected_fusion_cars_count += detected_fusion_cars
                    
                else:    # depricated
                    sub_image = image[40:1000, 200:1160]
                    elapsed, draw_time = activate_yolo(sub_image, detection_treshold, None, draw, elapsed_time, False)
                total_time += elapsed
            else:
                prev_pixels_radar =  pixels_radar2
                time_load_img_1 = time.time()
                file_path_img = dir_path + files_named + str(real_num) + '.jpg'
                image = cv2.imread(file_path_img)
                time_load_img_2 = time.time()
                readout_load_img += time_load_img_2 - time_load_img_1
        
                if mode == 0:
                    #elapsed, draw_time = activate_yolo(image, detection_treshold, pixels_radar2, draw, elapsed_time, draw_pts)
                    width = 640
                    height = 640
                    image_copy = deepcopy(image)
                    elapsed, draw_time, rdzen, fuzja, dzieci = yolo_core_and_children(model_core, model_child, image, detection_treshold, pixels_radar2, width, height, bias, draw, elapsed_time, draw_pts, [0, 2])
                    _, boxes_x, detected_classes_x, confidences_x = yolo_single_acivation(model_test, detection_treshold, image_copy, [0, 2])
                    if draw_x:
                        image_copy = draw_detect_info(image_copy, boxes_x, detected_classes_x, confidences_x, 'white', 1, 1)
                        width = 1280
                        height = 720

                        # Resize the image
                        resized_image = cv2.resize(image_copy, (width, height))
                        image_displayed = "image num {}".format(real_num)
                        cv2.putText(resized_image, image_displayed, (800, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)
                        cv2.imshow("image_x", resized_image)
                        cv2.waitKey(elapsed_time)
                    child_check += len(dzieci[2])
                    if label_mode:
                        cars, people, movement = label.label_photo_return(real_num)
                    else:
                        label_filename = dir_path + f"label_{real_num}.txt"
                        cars, people, movement = label.read_label_return(label_filename)
                    
                    people_count += people
                    car_count += cars
                    movement_real, Confidence_metric_x, Confidence_metric_core, Confidence_metric_fusion, Detected_x, Detected_core, Detected_fusion, conf_people, conf_cars = calculate_values(confidences_x, detected_classes_x,
                                                                                                                                                                                                rdzen[2], rdzen[1], fuzja[2], fuzja[1],
                                                                                                                                                                                                cars, people, movement, movement_real,
                                                                                                                                                                                                conf_people, conf_cars)
                    x_version_conf_people, x_version_conf_cars = Confidence_metric_x
                    core_version_conf_people, core_version_conf_cars = Confidence_metric_core
                    fusion_version_conf_people, fusion_version_conf_cars = Confidence_metric_fusion
                    
                    detected_x_people, detected_x_cars = Detected_x
                    detected_core_people, detected_core_cars = Detected_core
                    detected_fusion_people, detected_fusion_cars = Detected_fusion
                    
                    x_version_conf_people_count += x_version_conf_people
                    x_version_conf_cars_count += x_version_conf_cars
                    core_version_conf_people_count += core_version_conf_people
                    core_version_conf_cars_count += core_version_conf_cars
                    fusion_version_conf_people_count += fusion_version_conf_people
                    fusion_version_conf_cars_count += fusion_version_conf_cars
                    
                    detected_x_people_count += detected_x_people
                    detected_x_cars_count += detected_x_cars
                    detected_core_people_count += detected_core_people
                    detected_core_cars_count += detected_core_cars
                    detected_fusion_people_count += detected_fusion_people
                    detected_fusion_cars_count += detected_fusion_cars
                    
                elif mode == 1:  # depricated
                    bias = min(60, bias)
                    #sub_image = image[240:880, 300:940]
                    #elapsed, draw_time = activate_yolo(sub_image, detection_treshold, pixels_radar2, draw, elapsed_time, draw_pts)
                    width = 960
                    height = 960
                    image_copy = deepcopy(image)
                    elapsed, draw_time, rdzen, fuzja, dzieci = yolo_core_and_children(model_core, model_child, image, detection_treshold, pixels_radar2, width, height, bias, draw, elapsed_time, draw_pts, [0, 2])
                    _, boxes_x, detected_classes_x, confidences_x = yolo_single_acivation(model_test, detection_treshold, image_copy, [0, 2])
                    if draw_x:
                        image_copy = draw_detect_info(image_copy, boxes_x, detected_classes_x, confidences_x, 'white', 1, 1)
                        width = 1280
                        height = 720

                        # Resize the image
                        resized_image = cv2.resize(image_copy, (width, height))
                        image_displayed = "image num {}".format(real_num)
                        cv2.putText(resized_image, image_displayed, (800, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)
                        cv2.imshow("image_x", resized_image)
                        cv2.waitKey(elapsed_time)
                    child_check += len(dzieci[2])
                    if label_mode:
                        cars, people, movement = label.label_photo_return(real_num)
                    else:
                        label_filename = dir_path + f"label_{real_num}.txt"
                        cars, people, movement = label.read_label_return(label_filename)
                    
                    
                    people_count += people
                    car_count += cars
                    movement_real, Confidence_metric_x, Confidence_metric_core, Confidence_metric_fusion, Detected_x, Detected_core, Detected_fusion, conf_people, conf_cars = calculate_values(confidences_x, detected_classes_x,
                                                                                                                                                                                                rdzen[2], rdzen[1], fuzja[2], fuzja[1],
                                                                                                                                                                                                cars, people, movement, movement_real,
                                                                                                                                                                                                conf_people, conf_cars)
                    x_version_conf_people, x_version_conf_cars = Confidence_metric_x
                    core_version_conf_people, core_version_conf_cars = Confidence_metric_core
                    fusion_version_conf_people, fusion_version_conf_cars = Confidence_metric_fusion
                    
                    detected_x_people, detected_x_cars = Detected_x
                    detected_core_people, detected_core_cars = Detected_core
                    detected_fusion_people, detected_fusion_cars = Detected_fusion
                    
                    x_version_conf_people_count += x_version_conf_people
                    x_version_conf_cars_count += x_version_conf_cars
                    core_version_conf_people_count += core_version_conf_people
                    core_version_conf_cars_count += core_version_conf_cars
                    fusion_version_conf_people_count += fusion_version_conf_people
                    fusion_version_conf_cars_count += fusion_version_conf_cars
                    
                    detected_x_people_count += detected_x_people
                    detected_x_cars_count += detected_x_cars
                    detected_core_people_count += detected_core_people
                    detected_core_cars_count += detected_core_cars
                    detected_fusion_people_count += detected_fusion_people
                    detected_fusion_cars_count += detected_fusion_cars
                    
                else:   # depricated
                    sub_image = image[40:1000, 200:1160]
                    elapsed, draw_time = activate_yolo(sub_image, detection_treshold, pixels_radar2, draw, elapsed_time, draw_pts)
                total_time += elapsed
            
            
            
            if (this_frame_is_movement or last_frame_was_movement ) and movement > 0:
                detected_motion += 1
            last_frame_was_movement = this_frame_is_movement
            time_loop_b = time.time()
            time_loop = time_loop_b - time_load_txt_1
            total_time_loop += time_loop
            
            if draw:
                total_draw_time += draw_time
            
            real_num+=1

    
    img_count = file_count/(2+labels_exist)
    avg_time = total_time/img_count
    

    
    print('x = ', x)
    print('y = ', y)


time2 = time.time()
time_diff = time2 - time1
print('time - main loop: ', time_diff)
print('time - read txt: ', readout_load_txt)
print('time - read img: ', readout_load_img)
print('time - draw', total_draw_time)
print('time - read txt (average): ', f"{readout_load_txt/img_count*1000:.2f}")
print('time - read img (average): ', f"{readout_load_img/img_count*1000:.2f}")
print('time - draw time (average)', f"{total_draw_time/img_count*1000:.2f}")

print('average time_yolo =', f"{avg_time*1000:.2f}")
print('total_time_yolo =', f"{total_time*1000:.2f}")
avg_loop = total_time_loop/img_count
print('time -- typical loop = ', avg_loop)

# read the image
#image = cv2.imread("car_detect_day_2_nautical_c_1615_51.jpg")
#image = cv2.imread("test_field_e_2.jpg")

avg_point_count = point_count * (2+labels_exist) /file_count
avg_point_count_red = point_count_red * (2+labels_exist) / file_count
print('point count =', point_count)
print("Average num of points", avg_point_count)
print('point count (reduced) =', point_count_red)
print("Average num of points (reduced)", avg_point_count_red)

avg_point_count_red2 = point_count_red2 * (2+labels_exist) / file_count
print('point count (reduced2) =', point_count_red2)
print("Average num of points (reduced2)", avg_point_count_red2)

print('Max pts count: ', max_pts1, "reduced: ", max_pts2, "limited further", max_pts3)


print('final metrics ======================================== \n')
movement_ratio = detected_motion/movement_real
print('image count =', img_count)
print('Movement detected = ', detected_motion)
print('Movement ratio = ', movement_ratio, '\n')

# :.2f
if conf_cars > 0:
    print('Number of consensus car detections = ', conf_cars)
    print('Sum of all car detection confidences: \n')
    print(' x =', x_version_conf_cars_count)
    print(' core = ', core_version_conf_cars_count)
    print(' fusion = ', fusion_version_conf_cars_count)
    print('Average confidences in consensus: \n')
    print(' x =', f"{x_version_conf_cars_count/conf_cars:.3f}")
    print(' core = ', f"{core_version_conf_cars_count/conf_cars:.3f}")
    print(' fusion = ', f"{fusion_version_conf_cars_count/conf_cars:.3f}", "\n")

if conf_people > 0:
    print('Number of consensus people detections = ', conf_people)
    print('Sum of all people detection confidences: \n')
    print(' x =', x_version_conf_people_count)
    print(' core = ', core_version_conf_people_count)
    print(' fusion = ', fusion_version_conf_people_count)
    print('Average confidences in consensus: \n')
    print(' x =', f"{x_version_conf_people_count/conf_people:.3f}")
    print(' core = ', f"{core_version_conf_people_count/conf_people:.3f}")
    print(' fusion = ', f"{fusion_version_conf_people_count/conf_people:.3f}", "\n")
    

print('Number of car detections:')
print("x = ", detected_x_cars_count)
print("core = ", detected_core_cars_count)
print("fusion =", detected_fusion_cars_count)
print("labeled = ", car_count, "\n")

print('Number of people detections:')
print("x = ", detected_x_people_count)
print("core = ", detected_core_people_count)
print("fusion =", detected_fusion_people_count)
print("labeled = ", people_count, '\n')

print('child_check =', child_check)
print('check counter', check_counter)
print('conf cars', conf_cars)
