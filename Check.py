import os,shutil

path1= "C:/ml_project/ObjectCategories_sets/training_set"
path2 = "C:/ml_project/ObjectCategories_sets/test_set"
path="C:/ml_project/ObjectCategories_sets/TrainingSet"
fileName = os.listdir(path2)




for subdir ,_, files in os.walk(path2) :
      if (len(files) > 11) :
               newpath=path+"/"+os.path.basename(subdir)
               os.mkdir(newpath)
               for sub,_,files in os.walk(path1) :
                   if os.path.basename(subdir) in sub :
                            for file in files :
                                oldpath = sub +"/"+file
                                shutil.copy(oldpath,newpath)



