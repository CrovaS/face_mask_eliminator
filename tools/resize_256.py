from PIL import Image
import glob
import os
import string
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--path',dest='path')
path=parser.parse_args().path
target="dataset/resize_dataset"

if not os.path.isdir(target):
  os.mkdir(target)
  
#Resize the image on path into 256*256
imgs=glob.glob(path+"/*")
i=0
for image in imgs:
  if i%100==0:
    print(i)
  Img=Image.open(image)
  name=image.split("/")
  target_image = target+"/"+name[-1]
  # resize 할 이미지 사이즈 
  resize_image = Img.resize((256, 256))
  # 저장할 파일 Type : JPEG, PNG 등 
  # 저장할 때 Quality 수준 : 보통 95 사용 
  resize_image.save(target_image, "PNG", quality=95 )
  i+=1
