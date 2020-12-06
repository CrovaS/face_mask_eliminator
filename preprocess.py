from PIL import Image
import glob
import argparse
import os
from random import shuffle

parser=argparse.ArgumentParser()
parser.add_argument('--path',dest='path')
path=parser.parse_args().path
target_path="generative_inpainting/dataset"
target=["train", "val"]

if not os.path.isdir(target_path):
    os.mkdir(target_path)

for folder in target:
    if not os.path.isdir(target_path+"/"+folder):
        os.mkdir(target_path+"/"+folder)

imgs=glob.glob(path+"/*.png")
i=0
for image in imgs:
    k=i/len(imgs)*100
    if k%10==0:
        print(k,"%")
    folder_img, file_img = os.path.split(image)
    source_image=image
    if (i%10<8):
        target_image=target_path+"/"+target[0]+"/"+file_img
    else :
        target_image=target_path+"/"+target[1]+"/"+file_img
    Img=Image.open(source_image)
    resize_image=Img.resize((256,256))
    resize_image.save(target_image,"PNG",quality=95)
    i+=1
print(100,"%")


folder_path="generative_inpainting/dataset"
train_filename="generative_inpainting/dataset/train_shuffled.flist"
validation_filename="generative_inpainting/dataset/validation_shuffled.flist"
is_shuffled=1
# get the list of directories
##dirs = os.listdir(args.folder_path)
dirs_name_list = ["train","val"]

# make 2 lists to save file paths
training_file_names = []
validation_file_names = []

# print all directory names
i=0
for dir_item2 in dirs_name_list:
    # modify to full path -> directory
        
    dir_item = folder_path + "/" + dir_item2
    folder = os.listdir(dir_item)
    for item in folder:
        item = dir_item2 + "/" + item
        if i==0:
            training_file_names.append(item)
        else :
            validation_file_names.append(item)
    i=1

# print all file paths
for i in training_file_names:
    print(i)
for i in validation_file_names:
    print(i)

# This would print all the files and directories

# shuffle file names if set
if is_shuffled == 1:
    shuffle(training_file_names)
    shuffle(validation_file_names)

# make output file if not existed
#if not os.path.exists(args.train_filename):
 # os.mknod(args.train_filename)

  # if not os.path.exists(args.validation_filename):
   # os.mknod(args.validation_filename)

# write to file
fo = open(train_filename, "w")
fo.write("\n".join(training_file_names))
fo.close()

fo = open(validation_filename, "w")
fo.write("\n".join(validation_file_names))
fo.close()

# print process
print("Written file is: ", train_filename, ", is_shuffle: ", is_shuffled)
