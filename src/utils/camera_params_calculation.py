"""
float M_PI = radians(180.0);

in vec4 gl_FragCoord;

uniform vec3 cameraPos;
uniform vec3 camera_target;

int SCR_WIDTH = 1800;
int SCR_HEIGHT = 1000;

vec3 camera_up = vec3(0.0, 1.0, 0.0);

float camera_fovy =  45.0;
float focalHeight = 1.0;
float aspect = float(SCR_WIDTH)/float(SCR_HEIGHT);
float focalWidth = focalHeight * aspect; //Height * Aspect ratio
float focalDistance = focalHeight/(2.0 * tan(camera_fovy * M_PI/(180.0 * 2.0)));

float SMALLEST_DIST = 1e-4;
float FLT_MAX =  3.402823466e+38;
float t = FLT_MAX;

out vec4 color;

struct World{
    vec3 bgcolor;
};

struct Material{
    vec3 color;
};

struct Ray {
        vec3 origin;
        vec3 direction;
};

struct Sphere {
        vec3 origin;
        float radius;
        Material m;
};

bool intersect(Ray r, Sphere s);

void main() {

    World world;
    world.bgcolor = vec3(0.28, 0.28, 0.28);
    vec3 line_of_sight = camera_target - cameraPos;
    vec3 w = -normalize(line_of_sight);
    vec3 u = normalize(cross(camera_up, w));
    vec3 v = normalize(cross(w, u));
    float i = gl_FragCoord.x;
    float j = gl_FragCoord.y;
    vec3 dir = vec3(0.0, 0.0, 0.0);
	dir += -w * focalDistance;
	float xw = aspect*(i - SCR_WIDTH/2.0 + 0.5)/SCR_WIDTH;
	float yw = (j - SCR_HEIGHT/2.0 + 0.5)/SCR_HEIGHT;
	dir += u * xw;
	dir += v * yw;
    Ray r;
    r.origin = cameraPos;
    r.direction = normalize(dir);
    ...
    ...
}

The method to calculate ray equation is above.
However, it is doing a lot of repeated computations on the GPU.
I have fixed screen dimensions. Now, I can precompute alot of these values
and directly use them. Below is a small script that does that.
"""

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
