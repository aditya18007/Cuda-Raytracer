//
// Created by aditya on 09/04/22.
//

#ifndef HELIOS_OBJECT_LOADER_H
#define HELIOS_OBJECT_LOADER_H

#include <assimp/Importer.hpp>      // C++ importer interface
#include <assimp/scene.h>           // Output data structure
#include <assimp/postprocess.h>
#include <vector>
#include <glm/glm.hpp>
#define MAX_BONE_INFLUENCE 4

struct Vertex {
	glm::vec3 Position;
	glm::vec3 Normal;
};

struct Mesh_Positions{
	int start_vertices;
	int num_vertices;
	int start_indices;
	int num_indices;
};


class Mesh {
public:
	// mesh Data
	std::vector<Vertex> vertices;
	std::vector<unsigned int> indices;
public:
	Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices)
	{
		this->vertices = vertices;
		this->indices = indices;
	}
	const std::vector<Vertex>& get_vertices() const{
		return this->vertices;
	}
	
	const std::vector<unsigned int>& get_indices() const{
		return this->indices;
	}
};

class Object_Loader {
	std::vector<Mesh> meshes;
	const aiScene* m_scene;
public:
	Object_Loader(const std::string& object_name );
	const aiScene* get_scene();
	void processNode(aiNode *node, const aiScene *scene);
	Mesh processMesh(aiMesh *mesh, const aiScene *scene);
	 std::vector<Mesh>& get_meshes() {
		return meshes;
	}
};


#endif //HELIOS_OBJECT_LOADER_H
