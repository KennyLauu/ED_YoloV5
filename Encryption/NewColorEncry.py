'''
NOT USED - RESERVED 
AUTHOR: LONGBEFER
'''
import numpy as np

def NewColorEncry(img:np):
    w, h, z = img.shape
    s = [w, h, z]
    img = img.astype(np.float64)
    img 