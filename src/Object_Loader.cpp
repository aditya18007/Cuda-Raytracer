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
	m_scene = importer.ReadFile( obj_location, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_CalcTangentSpace);
	if (m_scene == nullptr){
		std::cout << "Failed to Load Object\n" << std::endl;
		exit(-1);
	}
	processNode(m_scene->mRootNode, m_scene);
	std::cout << meshes.size() << std::endl;
}

const aiScene *Object_Loader::get_scene() {
	return m_scene;
}

void Object_Loader::processNode(aiNode *node, const aiScene *scene) {
	// process each mesh located at the current node
	for(unsigned int i = 0; i < node->mNumMeshes; i++)
	{
		// the node object only contains indices to index the actual objects in the scene.
		// the scene contains all the data, node is just to keep stuff organized (like relations between nodes).
		aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
		meshes.push_back(processMesh(mesh, scene));
	}
	// after we've processed all of the meshes (if any) we then recursively process each of the children nodes
	for(unsigned int i = 0; i < node->mNumChildren; i++)
	{
		processNode(node->mChildren[i], scene);
	}
}

Mesh Object_Loader::processMesh(aiMesh *mesh, const aiScene *scene) {
	std::vector<Vertex> vertices;
	std::vector<unsigned int> indices;
	
	
	// walk through each of the mesh's vertices
	for(unsigned int i = 0; i < mesh->mNumVertices; i++)
	{
		Vertex vertex;
		glm::vec3 vector; // we declare a placeholder vector since assimp uses its own vector class that doesn't directly convert to glm's vec3 class so we transfer the data to this placeholder glm::vec3 first.
		// positions
		vector.x = mesh->mVertices[i].x;
		vector.y = mesh->mVertices[i].y;
		vector.z = mesh->mVertices[i].z;
		vertex.Position = vector;
		// normals
		if (mesh->HasNormals())
		{
			vector.x = mesh->mNormals[i].x;
			vector.y = mesh->mNormals[i].y;
			vector.z = mesh->mNormals[i].z;
			vertex.Normal = vector;
		}
		
		vertices.push_back(vertex);
	}
	// now wak through each of the mesh's faces (a face is a mesh its triangle) and retrieve the corresponding vertex indices.
	for(unsigned int i = 0; i < mesh->mNumFaces; i++)
	{
		aiFace face = mesh->mFaces[i];
		// retrieve all indices of the face and store them in the indices vector
		for(unsigned int j = 0; j < face.mNumIndices; j++)
			indices.push_back(face.mIndices[j]);
	}
	// process materials
	

	// return a mesh object created from the extracted mesh data
	return Mesh(vertices, indices);
}
