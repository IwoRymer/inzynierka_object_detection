import numpy as np

from timeit import default_timer as timer

start = timer()

# Save timestamp

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

x = [4, 6, 6, 6, 2]
y = [3, 3, 1, 4, 6]
limit = 2
distances = np.zeros((5, 5))
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

# Save timestamp
end = timer()

print(end - start)
print(distances)
print(neighbours)
print(new_pts)
print('x= ', new_x, ' y= ', new_y)


