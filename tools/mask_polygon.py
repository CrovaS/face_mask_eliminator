import os
import sys
import random
import argparse
import numpy as np
from PIL import Image, ImageFile, ImageDraw, ImagePath
import glob
import math
import json
import json

def cli():
    parser = argparse.ArgumentParser(description='Wear a face mask in the given picture.')
    parser.add_argument('pic_path', help='Picture path.')
    parser.add_argument('--show', action='store_true', help='Whether show picture with mask or not.')
    parser.add_argument('--model', default='hog', choices=['hog', 'cnn'], help='Which face detection model to use.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--black', action='store_true', help='Wear black mask')
    group.add_argument('--blue', action='store_true', help='Wear blue mask')
    group.add_argument('--red', action='store_true', help='Wear red mask')
    args = parser.parse_args()

    pic_path = args.pic_path
    if not os.path.exists(args.pic_path):
        print(f'Picture {pic_path} not exists.')
        sys.exit(1)

    if args.black:
        mask_path = BLACK_IMAGE_PATH
    elif args.blue:
        mask_path = BLUE_IMAGE_PATH
    elif args.red:
        mask_path = RED_IMAGE_PATH
    else:
        mask_path = DEFAULT_IMAGE_PATH

    FaceMasker(pic_path, mask_path, args.show, args.model).mask()

def create_mask(image_path, path, i):
    pic_path = image_path
    if i==1:
      mask_path="dataset/mask_image/blue-mask-2.png"
    if i==2:
      mask_path="dataset/mask_image/dentala0.png"
    if i==3:
      mask_path="dataset/mask_image/bluemask.png"
    if i==4:
      mask_path="dataset/mask_image/kf94.png"
    if i==5:
      mask_path="dataset/mask_image/kf94.png"
    if i==6:
      mask_path="dataset/mask_imagebluemaskc0.png"
    if i==7:
      mask_path="dataset/mask_image/default-mask.png"
    show = False
    model = "hog"
    return FaceMasker(pic_path, mask_path, path,show, model).mask()

def rot_pix(x,y,x_mid,y_mid,a):
    a=a*math.pi/180
    xr=(x-x_mid)*math.cos(a)-(y-y_mid)*math.sin(a)+x_mid
    yr=(x-x_mid)*math.sin(a)+(y-y_mid)*math.cos(a)+y_mid
    return xr,yr

class FaceMasker:
    KEY_FACIAL_FEATURES = ('nose_bridge', 'chin')

    def __init__(self, face_path, mask_path, path, show=False,model='hog'):
        self.face_path = face_path
        self.mask_path = mask_path
        self.show = show
        self.model = model
        self._face_img: ImageFile = None
        self._mask_img: ImageFile = None
        self.path=path

    def mask(self):
        import face_recognition

        face_image_np = face_recognition.load_image_file(self.face_path)
        face_locations = face_recognition.face_locations(face_image_np, model=self.model)
        face_landmarks = face_recognition.face_landmarks(face_image_np, face_locations)
        self._face_img = Image.fromarray(face_image_np)
        self._mask_img = Image.open(self.mask_path)

        found_face = False
        for face_landmark in face_landmarks:
            # check whether facial features meet requirement
            skip = False
            for facial_feature in self.KEY_FACIAL_FEATURES:
                if facial_feature not in face_landmark:
                    skip = True
                    break
            if skip:
                continue

            # mask face
            found_face = True
            k= self._mask_face(face_landmark)

        if found_face:
            if self.show:
                self._face_img.show()
            # save
            self._save()
        else:
            print('Found no face.')
        if found_face:
          return k
        else:
          return 0
    
    def _mask_face(self, face_landmark: dict):
        nose_bridge = face_landmark['nose_bridge']
        nose_point = nose_bridge[len(nose_bridge) * 1 // 4]
        nose_v = np.array(nose_point)

        chin = face_landmark['chin']
        chin_len = len(chin)
        chin_bottom_point = chin[chin_len // 2]
        chin_bottom_v = np.array(chin_bottom_point)
        chin_left_point = chin[chin_len // 8]
        chin_right_point = chin[chin_len * 7 // 8]

        # split mask and resize
        width = self._mask_img.width
        height = self._mask_img.height
        width_ratio = 1.2
        new_height = int(np.linalg.norm(nose_v - chin_bottom_v)*1.1)

        #MASK Polygon Rate
        left_high_h=3.5/12.38
        left_high_w=1.69/19.85
        left_low_h=11.72/12.38
        left_low_w=2.39/19.85
        right_high_h=3.5/12.38
        right_high_w=17.95/19.85
        right_low_h=11.25/12.38
        right_low_w=18.16/19.85

        middle_w=1/2
        middle_h=0.01

        # left
        mask_left_img = self._mask_img.crop((0, 0, width // 2, height))
        mask_left_width = self.get_distance_from_point_to_line(chin_left_point, nose_point, chin_bottom_point)
        mask_left_width = int(mask_left_width * width_ratio)
        mask_left_img = mask_left_img.resize((mask_left_width, new_height))

        left_high=(mask_left_width*left_high_w,new_height*left_high_h)
        left_low=(mask_left_width*left_low_w,new_height*left_low_h)


        # right
        mask_right_img = self._mask_img.crop((width // 2, 0, width, height))
        mask_right_width = self.get_distance_from_point_to_line(chin_right_point, nose_point, chin_bottom_point)
        mask_right_width = int(mask_right_width * width_ratio)
        mask_right_img = mask_right_img.resize((mask_right_width, new_height))

        right_high=(mask_right_width*right_high_w+mask_left_img.width,new_height*right_high_h)
        right_low=(mask_right_width*right_low_w+mask_left_img.width,new_height*right_low_h)

        # merge mask
        size = (mask_left_img.width + mask_right_img.width, new_height)
        mask_img = Image.new('RGBA', size)
        mask_img.paste(mask_left_img, (0, 0), mask_left_img)
        mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

        middle_high=(mask_left_width, new_height*middle_h)
        middle_low=(mask_left_width, new_height)
        
        # rotate mask
        angle = np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])
        rotated_mask_img = mask_img.rotate(angle, expand=True)

        middle_high=rot_pix(middle_high[0],middle_high[1],size[0]//2,size[1]//2,angle)
        middle_low=rot_pix(middle_low[0],middle_low[1],size[0]//2,size[1]//2,angle)
        right_high=rot_pix(right_high[0],right_high[1],size[0]//2,size[1]//2,angle)
        right_low=rot_pix(right_low[0],right_low[1],size[0]//2,size[1]//2,angle)
        left_high=rot_pix(left_high[0],left_high[1],size[0]//2,size[1]//2,angle)
        left_low=rot_pix(left_low[0],left_low[1],size[0]//2,size[1]//2,angle)

        # calculate mask location
        center_x = (nose_point[0] + chin_bottom_point[0]) // 2
        center_y = int(((nose_point[1] + chin_bottom_point[1]) // 2)*1.1)

        offset = mask_img.width // 2 - mask_left_img.width
        radian = angle * np.pi / 180
        box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
        box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2

        # add mask
        self._face_img.paste(mask_img, (box_x, box_y), mask_img)
        if (middle_low[1]+box_y)>255:
          middle_low_y=255
        else :
          middle_low_y=middle_low[1]+box_y
        
        if (right_low[1]+box_y)>255:
          right_low_y=255
        else:
          right_low_y=right_low[1]+box_y
        if (left_low[1]+box_y)>255:
          left_low_y=255
        else:
          left_low_y=left_low[1]+box_y
        middle_high=((int)(middle_high[0]+box_x),(int)(middle_high[1]+box_y))
        middle_low=((int)(middle_low[0]+box_x),(int)(middle_low_y))
        right_high=((int)(right_high[0]+box_x),(int)(right_high[1]+box_y))
        right_low=((int)(box_x+right_low[0]),(int)(right_low_y))
        left_high=((int)(box_x+left_high[0]),(int)(box_y+left_high[1]))
        left_low=((int)(box_x+left_low[0]),(int)(left_low_y))
        a=[left_high,middle_high,right_high,right_low,middle_low,left_low]

        
        poly=Image.new('RGBA',self._face_img.size)
        pdraw=ImageDraw.Draw(poly)
        pdraw.polygon(a,fill ="blue", outline ="blue")
        #self._face_img.paste(poly,mask=poly)
        k=(left_high,middle_high,right_high,right_low,middle_low,left_low)
        return k

    def _save(self):
        path_splits = os.path.split(self.face_path)
        new_face_path =  self.path + path_splits[1]
        self._face_img.save(new_face_path)
        print(f'Save to {new_face_path}')

    @staticmethod
    def get_distance_from_point_to_line(point, line_point1, line_point2):
        distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
                          (line_point1[0] - line_point2[0]) * point[1] +
                          (line_point2[0] - line_point1[0]) * line_point1[1] +
                          (line_point1[1] - line_point2[1]) * line_point1[0]) / \
                   np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
                           (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
        return int(distance)

if __name__ == '__main__':
    images = glob.glob('dataset/resize_dataset/*')
    print(len(images))
    if (os.path.isdir('facial_mask/train')):
        os.rmdir('facial_mask/train')
    if (os.path.isdir('facial_mask/val')):
        os.rmdir('facial_mask/val')
    #os.mkdir('facial_mask')
    os.mkdir('facial_mask/train')
    os.mkdir('facial_mask/val')
    train_dict=dict()
    val_dict=dict()
    i=0
    for fname in images:
      if i%2000<1800:
        element="facial_mask/train/"
      else:
        element="facial_mask/val/"
    
      if i<1500:
        k=create_mask(fname,element,1)
      if i>=1500 and i<3000:
        k=create_mask(fname,element,2)
      if i>=3000 and i<4500:
        k=create_mask(fname,element,3)
      if i>=4500 and i<6000:
        k=create_mask(fname,element,4)
      if i>=6000 and i<7500:
        k=create_mask(fname,element,5)
      if i>=7500 and i<9000:
        k=create_mask(fname,element,6)
      if i>=9000 and i<10000:
        k=create_mask(fname,element,7)
      if k:
        lh,mh,rh,rl,ml,ll=k
        path_splits = os.path.split(fname)
        xarr=[]
        xarr.append(lh[0])
        xarr.append(mh[0])
        xarr.append(rh[0])
        xarr.append(rl[0])
        xarr.append(ml[0])
        xarr.append(ll[0])
        yarr=[]
        yarr.append(lh[1])
        yarr.append(mh[1])
        yarr.append(rh[1])
        yarr.append(rl[1])
        yarr.append(ml[1])
        yarr.append(ll[1])

        if os.path.isfile(element + path_splits[1]):
          filename = path_splits[1]
          # processing json
          image = dict()
          image["fileref"]=""
          image["size"] = 6
          image["filename"] = filename
          image["base64_img_data"]=""
          image["file_attributes"]=dict()
          regions=dict()
          zero=dict()
          shape=dict()
          shape["name"]="polygon"
          shape["all_points_x"]=xarr
          shape["all_points_y"]=yarr
          zero["shape_attributes"]=shape
          zero["region_attributes"]=dict()
          regions["0"]=zero
          image["regions"]=regions
          path_splits = os.path.split(fname)
          if (i%2000<1800):
            train_dict[path_splits[1]]=image
          else :
            val_dict[path_splits[1]]=image
            print("I'm on")
      i+=1
    with open('facial_mask/train/via_region_data.json','w') as outfile:
        json.dump(train_dict,outfile)
    with open('facial_mask/val/via_region_data.json','w') as outfile:
        json.dump(val_dict,outfile)