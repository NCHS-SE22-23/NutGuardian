import os

folder = 'images/bird'

for count, filename in enumerate(os.listdir(folder)):
    src =f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
    dst =f"{folder}/bird{filename}"
         
    # rename() function will
    # rename all the files
    os.rename(src, dst)