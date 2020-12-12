from PIL import Image, ImageFile, ImageDraw, ImagePath, ImageEnhance

path="bluemask"
img=Image.open("bluemask.png")
enhancer1=ImageEnhance.Color(img)
for i in range(2):
    factor=1-0.7*i
    aug_image=enhancer1.enhance(factor)
    aug_name=path+"c"+str(i)+".png"
    aug_image.save(aug_name,"PNG",quality=95)

enhancer2=ImageEnhance.Brightness(img)
for j in range(2):
    factor=1-0.7*j
    aug_image=enhancer2.enhance(factor)
    aug_name=path+"a"+str(j)+".png"
    aug_image.save(aug_name,"PNG",quality=95)