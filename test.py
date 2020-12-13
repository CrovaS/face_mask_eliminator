from Mask_RCNN import test
import os

pid = os.fork()
if pid==0:
    os.chdir("Mask_RCNN")
    print(1)
    exec(open("test.py").read())
else:
    os.wait()
    print(2)
    os.chdir("generative_inpainting")
    exec(open("test.py").read())
