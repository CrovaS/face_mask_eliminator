from PIL import Image, ImageFile, ImageDraw, ImagePath, ImageEnhance
import numpy as np
import glob
import argparse
import os
from random import shuffle
import face_recognition

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

imgs=glob.glob(path+"/*")
i=0
for image in imgs:
    img=Image.open(image)
    k=i/len(imgs)*100
    if k%10==0:
        print(k,"%")
    s = os.path.splitext(image)
    s= os.path.split(s[0])
    file_img=s[1]
    source_image=image
    if (i%10<8):
        target_image=target_path+"/"+target[0]+"/"+file_img+".png"
    else :
        target_image=target_path+"/"+target[1]+"/"+file_img+".png"
    face_image_np=face_recognition.load_image_file(source_image)
    face_locations=face_recognition.face_locations(face_image_np,model="hog")
    face_landmarks=face_recognition.face_landmarks(face_image_np,face_locations)
    for face_landmark in face_landmarks:
        nose_bridge=face_landmark['nose_bridge']
        nose_point=nose_bridge[len(nose_bridge)*1//4]
        nose_v=np.array(nose_point)
        chin=face_landmark['chin']
        chin_len=len(chin)
        chin_bottom_point=chin[chin_len//2]
        lower=nose_point[1]+(chin_bottom_point[1]-nose_point[1])
        right=nose_point[0]+(chin_bottom_point[1]-nose_point[1])
        left=nose_point[0]-(chin_bottom_point[1]-nose_point[1])
        upper=nose_point[1]-(chin_bottom_point[1]-nose_point[1])
        cropped_img=img.crop((left,upper,right,lower))
        resize_image=cropped_img.resize((256,256))
        resize_image.save(target_image,"PNG",quality=95)

        enhancer1=ImageEnhance.Color(resize_image)
        for k in range(10):
            factor=1-0.05*k
            aug_image=enhancer1.enhance(factor)
            aug_name=target_image+"c"+str(k)+".png"
            aug_image.save(aug_name,"PNG",quality=95)
        
        enhancer2=ImageEnhance.Brightness(resize_image)
        for j in range(10):
            factor=1-0.05*j
            aug_image=enhancer2.enhance(factor)
            aug_name=target_image+"b"+str(j)+".png"
            aug_image.save(aug_name,"PNG",quality=95)

        
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
        item = "dataset/"+dir_item2 + "/" + item
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
