import os
obj = dict()

directory = "C:\ObjectCategories"
filename = os.listdir(directory)
for iter in range(len(filename)) :
    obj[iter] =  filename[iter]



