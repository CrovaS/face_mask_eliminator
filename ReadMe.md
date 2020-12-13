# Face Mask Eliminator
CS470 Final Project : Face Mask Eliminator
<img class="fit-picture"
     src="https://github.com/hmc0105/face_mask_eliminator/blob/master/mode.png"
     >

## [ About ]
<a href="https://drive.google.com/file/d/1BhTdI1iVxx1nCMz4IHTGdTHiwT73w_tq/view?usp=sharing">Project Paper</a>


<a href="https://colab.research.google.com/drive/1w7Md6DVNH9izUUBWfn-_L5OWoL8KrnXj?usp=sharing">Colab</a>
<a href="https://www.youtube.com/watch?v=X4KrHy1tkTY&t=2s">Colab Guide Video</a>

### Abstract on Project Paper
After COVID-19 breakout, it becomes hard to take a photo without the face mask to prevent the spread of the illness. 
To solve the problem, this paper will suggest about personal face mask eliminator. 
It consists of the semantic segmentation model and the inpainting model to recover one’s face. 
Specified training methods for model to be personalized will be explained on the paper.

## [ Develop Environment ]
- Tensorflow 1.1.15
- Keras 2.2.5


## [ SETUP ]
Tested in MacBook Pro 2019 13'<br>
Tested in Colab<br>
<br>
pip install keras==2.2.5<br>
pip install tensorflow==1.15.0<br>
pip install -q PyDrive<br>
pip install -r requirements.txt<br>
pip install git+https://github.com/JiahuiYu/neuralgym<br>
pip install face_recognition<br>
cd Mask_RCNN<br>
python setup.py install<br>
cd ../<br>

<br>
<a href="https://beausty23.tistory.com/59">Tensorflow 1.15.0 Install Guideline</a>

## [ Execution ]
Before the start, download the below things

* Mask RCNN : Save the file onto Mask_RCNN/logs
<a href="https://drive.google.com/file/d/1DZ1YmEiW39JBs3bn-KlUvZ1dhlOzdKEd/view?usp=sharing">https://drive.google.com/file/d/1DZ1YmEiW39JBs3bn-KlUvZ1dhlOzdKEd/view?usp=sharing</a>


* DeepFill v2 : Save the file onto generative_inpainting/pre_trained
<a href="https://drive.google.com/drive/folders/1jDDjRJPHR1MS20fujUF6hW_2Z3upQQhz?usp=sharing">https://drive.google.com/drive/folders/1jDDjRJPHR1MS20fujUF6hW_2Z3upQQhz?usp=sharing</a>

* Use your front-looking faces 20~25 figure, or You can use dataset folder
1. conda activate tensorflow

2. python preprocess.py --path folder_your_face

3. python train.py

4. python test.py : In here, your image with the face mask should be 256*256 size



## [OS X Execution Error in Encoding Solution]
"/Users/User_name/opt/anaconda3/envs/tensorflow/lib/python3.7/site-packages/keras/engine/saving.py"

<img class="encoding_error"
     src="https://github.com/hmc0105/face_mask_eliminator/blob/master/encoding_error.png"
     >


## [MASK R-CNN Training]

* Unzip the file and put into Mask_RCNN/dataset <br>
https://drive.google.com/file/d/1XIqZwpaYSw4SHCkCVYkaGpiYfl8htmxH/view?usp=sharing

<br>
<br>

cd Mask_RCNN<br>
python facial_mask.py train --datset=dataset/ -weights=coco<br>


## [ Library ]
- [Generative Image Inpainting](https://github.com/JiahuiYu/generative_inpainting)
- [Mask R-CNN](https://github.com/matterport/Mask_RCNN)
- [face_recognition](https://github.com/ageitgey/face_recognition)