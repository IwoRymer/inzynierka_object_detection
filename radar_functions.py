import numpy as np

def search(idx, pts, neighbours, visited):
    contacts = neighbours[idx]
    for _, point in enumerate(contacts):
        if not visited[point]:
            pts.append(point)
            visited[point] = True
            pts, visited = search(point, pts, neighbours, visited)
        pass
    pass
    return pts, visited


def limit_points(x, y, limit):

    distances = np.zeros((len(x), (len(x))))
    visited = [False for i in range(0, len(y))]

    for i in range(len(y)):
        for j in range(len(x)):
            distances[i][j] = (y[i]-y[j])**2 + (x[i] - x[j])**2
        

    neighbours = {}
    limit_sq = limit**2
    for i in range(len(y)):
        neighbour_lst = []
        for j in range(0, len(x)):
            if distances[i][j] <= limit_sq:
                neighbour_lst.append(j)
        neighbours[i] = neighbour_lst
    
    
    new_pts = {}
    for i in range(len(y)):
        if not visited[i]:
            pts_in_obj = []
            first = neighbours[i][0]
            pts_in_obj.append(first)
            visited[i] = True
            pts_in_obj, visited = search(first, pts_in_obj, neighbours, visited)
            new_pts[i] = pts_in_obj
        pass

    new_x = []
    new_y = []
    for key in new_pts:
        pts_list = new_pts[key]
        new_x_val = 0
        new_y_val = 0
        for idx in pts_list:
            new_x_val += x[idx]
            new_y_val += y[idx]
        new_x_val = new_x_val/len(pts_list)
        new_y_val = new_y_val/len(pts_list)
        new_x.append(new_x_val)
        new_y.append(new_y_val)

    return new_x, new_y


def detection_2_pixels(x_array, y_array, false_distance, camera_center_x) -> list[int]:
    pixels_movement = []
    for i in range(0, len(x_array)):
        x_ = x_array[i]
        y_ = y_array[i]
        obj_tan = x_/y_
        cam_x = false_distance * obj_tan
        if np.abs(cam_x) <= camera_center_x - 4:
            #cam_x = camera_center_x - 5
            pixels_movement.append(int(cam_x))
    return pixels_movement


def filter_pixels_after(pixels, limit=5):
    if pixels is not None and len(pixels) > 1:
        new_pixels = [pixels[0]]
        for i in range(1, len(pixels)):
            pixel = pixels[i]
            save = True
            for j in range(0, len(pixels)):
                if np.abs(pixel - pixels[j]) < limit and i != j:
                    save = False
            if save:
                new_pixels.append(pixel)
        return new_pixels
    else:
        return pixels


def filter_clutter_from_mask(x_mask, y_mask, x, y, limit=0.25):
    new_x = []
    new_y = []
    
    if x_mask != [] and y_mask != []:
        for i in range(0, len(x)):
            #dla kazdej detekcji
            detection_x = x[i]
            detection_y = y[i]
            
            save = True
            for j in range(0, len(x_mask)):
                distance = np.sqrt((detection_x-x_mask[j])**2  + (detection_x-x_mask[j])**2)
                if distance < limit:
                    save = False
            if save:
                new_x.append(detection_x)
                new_y.append(detection_y)
    return new_x, new_y