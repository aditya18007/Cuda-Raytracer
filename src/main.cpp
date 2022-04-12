#include "Application.h"
#include "Dimensions.h"
#include "Object_Loader.h"
#include <iostream>
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

int main(int, char**)
{
	Object_Loader loader("backpack.obj");
	Application a(WIDTH,HEIGHT );
	a.run(loader);
	
	
	return 0;
}