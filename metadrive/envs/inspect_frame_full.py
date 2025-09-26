from PIL import Image
import numpy as np
p = r'c:\Users\37945\OneDrive\Desktop\metadrive\metadrive\envs\bev_sequence_sss\frame_0066.png'
img = Image.open(p).convert('RGBA')
arr = np.array(img)
h,w = arr.shape[0], arr.shape[1]
# thresholds
green_pixels = []
blue_pixels = []
white_pixels = []
nonblack=0
for y in range(h):
    for x in range(w):
        r,g,b,a = arr[y,x,:]
        if r+g+b>10:
            nonblack+=1
        if g>180 and r<120 and b<120:
            green_pixels.append((x,y,(r,g,b,a)))
        if b>180 and r<120 and g<120:
            blue_pixels.append((x,y,(r,g,b,a)))
        if r>200 and g>200 and b>200:
            white_pixels.append((x,y,(r,g,b,a)))
print('image',w,h)
print('nonblack total', nonblack)
print('green count', len(green_pixels), 'blue count', len(blue_pixels), 'white count', len(white_pixels))
if green_pixels:
    gx = sum([p[0] for p in green_pixels])/len(green_pixels)
    gy = sum([p[1] for p in green_pixels])/len(green_pixels)
    print('green centroid', gx,gy, 'sample', green_pixels[:10])
if blue_pixels:
    bx = sum([p[0] for p in blue_pixels])/len(blue_pixels)
    by = sum([p[1] for p in blue_pixels])/len(blue_pixels)
    print('blue centroid', bx,by, 'sample', blue_pixels[:10])
if white_pixels:
    wx = sum([p[0] for p in white_pixels])/len(white_pixels)
    wy = sum([p[1] for p in white_pixels])/len(white_pixels)
    print('white centroid', wx,wy)
# print a small region around the green centroid

