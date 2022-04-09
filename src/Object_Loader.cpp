//
// Created by aditya on 09/04/22.
//

#include "Object_Loader.h"
#include <iostream>
Object_Loader::Object_Loader(const std::string& object_name) {
	std::string pwd(__FILE__);
	auto last = pwd.find_last_of('/');
	auto directory = pwd.substr(0, last);
	last = directory.find_last_of('/');
	directory = directory.substr(0, last);
	directory += "/assets/scenes/";
	auto obj_location = directory + object_name;
	
	Assimp::Importer importer;
	m_scene = importer.ReadFile( obj_location, aiProcess_Triangulate);
	if (m_scene == nullptr){
		std::cout << "Failed to Load Object\n" << std::endl;
	}
}
