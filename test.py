from Mask_RCNN import test
from generative_inpainting import test
import os

os.chdir("Mask_RCNN")
exec(open("test.py").read())

os.chdir("../generative_inpainting")
exec(open("test.py").read())