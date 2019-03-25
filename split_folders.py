import os,shutil

directory = "C:/ml_project/ObjectCategories"
dir2 = "C:/ml_project/ObjectCategories_sets"
os.mkdir(dir2)

fileName = os.listdir(directory)
path1= "C:/ml_project/ObjectCategories_sets/training_set"
path2 = "C:/ml_project/ObjectCategories_sets/test_set"

os.mkdir(path1)
for iter in fileName :
         os.mkdir(path1+"/"+iter)

os.mkdir(path2)
for iter in fileName :
       os.mkdir(path2+"/"+iter)

for subdir , dir , files in os.walk(directory) :
    split1 = int(0.8*len(files))
    split2=int(0.8*len(files))
    train_files = files[:split1]
    test_files = files[split2:]

    if train_files != [] :
        for file in train_files :
          oldpath = subdir + "/" + file
          for iter in fileName :
            if iter in subdir:
              newpath  = path1 + "/" + iter
          shutil.copy(oldpath,newpath)

    if test_files != [] :
        for file in test_files :
          oldpath = subdir + "/" + file
          for iter in fileName :
            if iter in subdir:
              newpath  = path2 + "/" + iter
          shutil.copy(oldpath,newpath)








