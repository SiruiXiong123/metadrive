from PIL import Image
import numpy as np
import sys
p = r'c:\Users\37945\OneDrive\Desktop\metadrive\metadrive\envs\bev_sequence_sss\frame_0066.png'
img = Image.open(p).convert('RGBA')
arr = np.array(img)
h,w = arr.shape[0], arr.shape[1]
coords = []
for y in range(0,h):
    for x in range(0,w):
        if (arr[y,x,:3].sum()>30):
            coords.append((x,y, tuple(arr[y,x,:])))
coords = coords[:500]
print('image size', w,h)
print('found', len(coords), 'non-black pixels (showing up to 200):')
for i,c in enumerate(coords[:200]):
    if i<50:
        print(c)
# find the brightest green-ish and blue-ish pixels
green = [(x,y,rgba) for x,y,rgba in coords if rgba[1]>200 and rgba[0]<120 and rgba[2]<120]
blue = [(x,y,rgba) for x,y,rgba in coords if rgba[2]>200 and rgba[0]<120 and rgba[1]<120]
white = [(x,y,rgba) for x,y,rgba in coords if rgba[0]>200 and rgba[1]>200 and rgba[2]>200]
print('sample green count', len(green), 'blue count', len(blue), 'white count', len(white))
print('some green pixels (x,y,rgba):', green[:20])
print('some blue pixels (x,y,rgba):', blue[:20])
print('some white pixels (x,y,rgba):', white[:20])
# print centroid of green pixels
if green:
    gx = sum([p[0] for p in green]) / len(green)
    gy = sum([p[1] for p in green]) / len(green)
    print('green centroid', gx,gy)
if blue:
    bx = sum([p[0] for p in blue]) / len(blue)
    by = sum([p[1] for p in blue]) / len(blue)
    print('blue centroid', bx,by)
if white:
    wx = sum([p[0] for p in white]) / len(white)
    wy = sum([p[1] for p in white]) / len(white)
    print('white centroid', wx,wy)
