import numpy as np 
import os 
import start as st 
import matplotlib.pyplot as plt 

# Function that go through all the files and append each in a list , then return the list 

def get_data():
    t_data = []
    data = os.listdir('Database')
    for folder in data:  
        temp_data = []  
        s_class = int(folder)
        files = os.listdir('Database' + '\\' + folder)
        for ff in files :
            s_file = 'Database' + '\\' + folder + '\\' + ff
            temp_data.append(st.get_features_file(s_file, s_class))
        temp_data = np.array(temp_data)
        t_data.append(temp_data)
    return t_data


# Function used just to get points from a given class 

def get_points_class(class_nb):
    t_data = []
    data = os.listdir('Database')
    files = os.listdir('Database' + '\\' + data[class_nb - 1])
    for ff in files:
        s_file = 'Database' + '\\' + data[class_nb - 1] + '\\' + ff
        points = st.get_points_file(s_file)
        t_data.append(points)
        break
    return np.array(t_data[0])

# Function used to plot a single signature 
            
def plot_point(pp):
    plt.plot(pp[:, 0], pp[:, 1], color = 'black')
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


