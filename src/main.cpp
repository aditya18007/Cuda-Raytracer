#include "Application.h"
#include "Dimensions.h"
#include "Object_Loader.h"
#include <iostream>
int main(int, char**)
{
	
	Object_Loader loader("backpack.obj");
//	Application a(WIDTH,HEIGHT );
//	a.run(loader);
	auto* scene = loader.get_scene();
	return 0;
}