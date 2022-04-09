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

#Simplifying xw calculation
x = 511 #example
xw = aspect*(x - width/2.0 + 0.5)/width
x_mul = aspect/width
xoff = ((0.5*aspect)/width) - (aspect/2)
xw2 = x_mul*(x) + xoff 
print(f"x_mul = {x_mul}")
print(f" xw offset = {xoff}")

#Simplifying yw calculation
y = 329 #example
yw = (y - height/2.0 + 0.5)/height
yoff =  (0.5/height)
yw2 = (y/height) - (0.5) + yoff
print(yw)
print(yw2)
print(f"yw offset = {yoff}")
