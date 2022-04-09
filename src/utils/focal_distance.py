import math
width = float(input('Enter Width : '))
height = float(input('Enter Height : '))
camera_fovy = 45.0
focal_height = 1.0
aspect =width/height
print(f"Aspect Ration = {aspect}")

focal_width = focal_height*aspect
focal_distance = focal_height/(2 * math.tan(camera_fovy *math.pi/360 ))
print(f"Focal Distance = {focal_distance}")