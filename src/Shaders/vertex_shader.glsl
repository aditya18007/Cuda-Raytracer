#version 460 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aTexCoord;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPos.x, aPos.y, 1.0, 1.0);
    TexCoord = vec2(aTexCoord.x, aTexCoord.y);
}