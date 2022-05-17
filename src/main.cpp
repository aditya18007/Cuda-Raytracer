#include "Application.h"
#include "Dimensions.h"
#include "Object_Loader.h"
#include <iostream>

int main(int argc, char** argv)
{
	if (argc < 2){
		std::cout << "enter model name\n";
		exit(-1);
	}
	std::string name(argv[1]);
	Object_Loader loader(name);
	Application a(WIDTH,HEIGHT );
	a.load_model(loader);
	a.run();
	return 0;
}