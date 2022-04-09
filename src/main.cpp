#include "Application.h"
#include "Dimensions.h"
#include "Object_Loader.h"
#include <iostream>
int main(int, char**)
{
	
	Object_Loader loader("sofa.obj");
	Application a(WIDTH,HEIGHT );
	a.run(loader);
	
	return 0;
}