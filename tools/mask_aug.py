from PIL import Image, ImageFile, ImageDraw, ImagePath, ImageEnhance
import glob

path="dataset/mask_image"

#Face Mask Image Augmentation on /tools/dataset/mask_image
mask_imgs=glob.glob(path+"/*")
for image in mask_imgs:
    name=image.split('/')
    img=Image.open(image)
    enhancer1=ImageEnhance.Color(img)
    for i in range(2):
        factor=1-0.1*i
        aug_image=enhancer1.enhance(factor)
        aug_name=path+"/c"+str(i)+name[-1]
        aug_image.save(aug_name,"PNG",quality=95)

    enhancer2=ImageEnhance.Brightness(img)
    for j in range(2):
        factor=1-0.1*j
        aug_image=enhancer2.enhance(factor)
        aug_name=path+"/b"+str(i)+name[-1]
        aug_image.save(aug_name,"PNG",quality=95)
