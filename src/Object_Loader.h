//
// Created by aditya on 09/04/22.
//

#ifndef HELIOS_OBJECT_LOADER_H
#define HELIOS_OBJECT_LOADER_H

#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/scene.h>           // Output data structure
#include <assimp/postprocess.h>
class Object_Loader {
	const aiScene* m_scene;
public:
	Object_Loader(const std::string& object_name );
};


#endif //HELIOS_OBJECT_LOADER_H
