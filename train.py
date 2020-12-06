from generative_inpainting import train
import os
os.chdir("generative_inpainting")
exec(open("generative_inpainting/train.py").read())
