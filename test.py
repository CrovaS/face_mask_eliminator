from Mask_RCNN import test
import os

pid = os.fork()
if pid==0:
    os.chdir("Mask_RCNN")
    exec(open("test.py").read())
else:
    os.wait()
    os.chdir("generative_inpainting")
    exec(open("test.py").read())
