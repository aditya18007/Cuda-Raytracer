#include "Application.h"
#include "Dimensions.h"
#include "Object_Loader.h"

int main(int, char**)
{
	Object_Loader loader("deer.model");
	Application a(WIDTH,HEIGHT );
	a.load_model(loader);
	a.run();
	return 0;
}