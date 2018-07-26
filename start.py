import numpy as np

# Function that return the distance between two points 
def dist(x, y):
    return np.sqrt(np.sum((x-y)**2))

#function that return the total distance 

def total_dist(tab):
    s = 0
    for j in range(1, len(tab)):
        s += dist(tab[j-1], tab[j])
        j += 1
    return s

#function that return the average velocity 

def velocity(axis, t_stamp):
    v = []
    for j in range(1, len(axis)):
        temp_v = np.abs((axis[j] - axis[j-1]) )/ (t_stamp[j] - t_stamp[j-1])
        v.append(temp_v)
    v = np.array(v)
    return np.mean(v)

# Extractring feature from a single file 

def get_features_file(ff, index = 0):
    s_data = []
    with open(ff) as file:
        file.readline()
        file.readline()
        for ff in file:
            temp = ff.split()
            temp = [int(jj) for jj in temp]
            temp = np.array(temp)
            s_data.append(temp)
    
    s_data = np.array(s_data)

    points = s_data[:, 0:2]

    t_dist = total_dist(points)

    s_width = np.max(points[:, 0]) - np.min(points[:, 0])

    s_height = np.max(points[:, 1]) - np.min(points[:, 1])

    sum_x = np.sum(points[:, 0])

    sum_y = np.sum(points[:, 1])

    a_pressure = np.mean(s_data[:, 3])

    x_velocity = velocity(s_data[:, 0], s_data[:, 2])

    y_velocity = velocity(s_data[:, 1], s_data[:, 2])
    
    return t_dist, s_width, s_height, sum_x, sum_y, a_pressure, x_velocity, y_velocity, index
    

# Fuction used just to get all the points x and y from a sig file 

def get_points_file(ff):
    s_data = []
    with open(ff) as file:
        file.readline()
        file.readline()
        for ff in file :
            temp = ff.split()
            temp = [int(jj) for jj in temp]
            temp = np.array(temp)
            s_data.append(temp[:2])
    return s_data    
       


























