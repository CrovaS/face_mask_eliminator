from PIL import Image
import glob
import os
#path="/content/drive/MyDrive/project/Dataset/img_align_celeba/img_align_celeba"
path="/content/sy"
target="/content/val"
if os.path.isdir(target):
  os.rmdir(target)
os.mkdir(target)
#%cd "/content/drive/MyDrive/project/Dataset/img_align_celeba/img_align_celeba"
%cd "/content/sy"
imgs=glob.glob("*.jpg")
i=0
for image in imgs:
  if i%100==0:
    print(i)
  source_image = path+"/"+image
  target_image = target+"/"+image
  image = Image.open(source_image)
  # resize 할 이미지 사이즈 
  resize_image = image.resize((256, 256))
  # 저장할 파일 Type : JPEG, PNG 등 
  # 저장할 때 Quality 수준 : 보통 95 사용 
  resize_image.save(target_image, "PNG", quality=95 )
  i+=1