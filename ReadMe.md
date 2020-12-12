# Face Mask Eliminator
CS470 Final Project : Face Mask Eliminator
<img class="fit-picture"
     src="https://github.com/hmc0105/face_mask_eliminator/blob/master/mode.png"
     >

## [ About ]
<a href="https://drive.google.com/file/d/1BhTdI1iVxx1nCMz4IHTGdTHiwT73w_tq/view?usp=sharing">Project Paper</a>

Highly recommended to work on Colab environment
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
pip install keras==2.2.5
pip install tensorflow==1.1.15
pip install -q PyDrive
pip install -r requirements.txt
python setup.py install
pip install git+https://github.com/JiahuiYu/neuralgym
pip install face_recognition

## [ Execution ]
Before the start, download the below things

* Mask RCNN : Save the file onto Mask_RCNN/logs
<a href="https://drive.google.com/file/d/1DZ1YmEiW39JBs3bn-KlUvZ1dhlOzdKEd/view?usp=sharing">https://drive.google.com/file/d/1DZ1YmEiW39JBs3bn-KlUvZ1dhlOzdKEd/view?usp=sharing</a>


* DeepFill v2 : Save the file onto generative_inpainting/pre_trained
<a href="https://drive.google.com/drive/folders/1jDDjRJPHR1MS20fujUF6hW_2Z3upQQhz?usp=sharing">https://drive.google.com/drive/folders/1jDDjRJPHR1MS20fujUF6hW_2Z3upQQhz?usp=sharing</a>

* Upload your front-looking faces 20~25 figure

1. python preprocess.py --path folder_your_face

2. python train.py

3. python test.py


## [ Library ]
- [Generative Image Inpainting](https://github.com/JiahuiYu/generative_inpainting)
- [Mask R-CNN](https://github.com/matterport/Mask_RCNN)
- [face_recognition](https://github.com/ageitgey/face_recognition)