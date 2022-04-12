#include "Application.h"
#include "Dimensions.h"
#include "Object_Loader.h"
#include <iostream>

int main(int, char**)
{
	Object_Loader loader("low_poly_car.obj");
	Application a(WIDTH,HEIGHT );
	a.load_model(loader);
	a.run();
	return 0;
}