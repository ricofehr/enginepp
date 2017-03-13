/*
* Wall class file
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#include "engine/universe/wall.h"

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <SOIL/SOIL.h>

#include <iostream>
#include <memory>

#include "engine/geometry/quad.h"

namespace engine {
namespace universe {

namespace {
GLuint vertexbuffer[6] = {0,0,0,0,0,0};
GLuint texturebuffer[6] = {0,0,0,0,0,0};

const GLfloat g_buffer_data[6][32] = {
   /* Position            Color              Texcoords */
{   /* Front */
    -1.0f, -0.4f,  1.0f,  1.0f, 1.0f, 1.0f,  0.0f,  0.0f,
     1.0f, -0.4f,  1.0f,  1.0f, 1.0f, 1.0f,  4.0f,  0.0f,
     1.0f,  0.4f,  1.0f,  1.0f, 1.0f, 1.0f,  4.0f,  4.0f,
    -1.0f,  0.4f,  1.0f,  1.0f, 1.0f, 1.0f,  0.0f,  4.0f,
},

{    /* Right */
     1.0f, -0.4f,  1.0f,  1.0f, 1.0f, 1.0f,  0.0f,  0.0f,
     1.0f, -0.4f, -1.0f,  1.0f, 1.0f, 1.0f,  4.0f,  0.0f,
     1.0f,  0.4f, -1.0f,  1.0f, 1.0f, 1.0f,  4.0f,  4.0f,
     1.0f,  0.4f,  1.0f,  1.0f, 1.0f, 1.0f,  0.0f,  4.0f,
},

{   /* Back */
    -1.0f,  0.4f, -1.0f,  1.0f, 1.0f, 1.0f,  0.0f,  0.0f,
     1.0f,  0.4f, -1.0f,  1.0f, 1.0f, 1.0f,  4.0f,  0.0f,
     1.0f, -0.4f, -1.0f,  1.0f, 1.0f, 1.0f,  4.0f,  4.0f,
    -1.0f, -0.4f, -1.0f,  1.0f, 1.0f, 1.0f,  0.0f,  4.0f,
},

{   /* Left */
    -1.0f, -0.4f, -1.0f,  1.0f, 1.0f, 1.0f,  0.0f,  0.0f,
    -1.0f, -0.4f,  1.0f,  1.0f, 1.0f, 1.0f,  4.0f,  0.0f,
    -1.0f,  0.4f,  1.0f,  1.0f, 1.0f, 1.0f,  4.0f,  4.0f,
    -1.0f,  0.4f, -1.0f,  1.0f, 1.0f, 1.0f,  0.0f,  4.0f,
},

{   /* Bottom (!floor is a wall object) */
    -1.0f, -0.4f, -1.0f,  1.0f, 1.0f, 1.0f,  0.0f,  0.0f,
     1.0f, -0.4f, -1.0f,  1.0f, 1.0f, 1.0f,  10.0f, 0.0f,
     1.0f, -0.4f,  1.0f,  1.0f, 1.0f, 1.0f,  10.0f, 10.0f,
    -1.0f, -0.4f,  1.0f,  1.0f, 1.0f, 1.0f,  0.0f,  10.0f,
},

{  /* Top (!roof is a wall object) */
    -1.0f,  0.4f,  1.0f,  1.0f, 1.0f, 1.0f,  0.0f,  0.0f,
     1.0f,  0.4f,  1.0f,  1.0f, 1.0f, 1.0f,  1.0f,  0.0f,
     1.0f,  0.4f, -1.0f,  1.0f, 1.0f, 1.0f,  1.0f,  1.0f,
    -1.0f,  0.4f, -1.0f,  1.0f, 1.0f, 1.0f,  0.0f,  1.0f,
},

};


const std::vector<glm::vec3> coords[6] = {
/* Front Border */
{
    {-1.0f,  0.4f,  1.1f},
    { 1.0f,  0.4f,  1.1f},
    { 1.0f, -0.4f,  1.1f},
    {-1.0f, -0.4f,  1.1f},

    {-1.0f,  0.4f,  1.0f},
    { 1.0f,  0.4f,  1.0f},
    { 1.0f, -0.4f,  1.0f},
    {-1.0f, -0.4f,  1.0f},
},
/* Right border */
{
    {1.0f,  0.4f, 1.0f},
    {1.1f,  0.4f, 1.0f},
    {1.1f, -0.4f, 1.0f},
    {1.0f, -0.4f, 1.0f},

    {1.0f,  0.4f, -1.0f},
    {1.1f,  0.4f, -1.0f},
    {1.1f, -0.4f, -1.0f},
    {1.0f, -0.4f, -1.0f},
},
/* Back border */
{
    {-1.0f,  0.4f, -1.0f},
    { 1.0f,  0.4f, -1.0f},
    { 1.0f, -0.4f, -1.0f},
    {-1.0f, -0.4f, -1.0f},

    {-1.0f,  0.4f, -1.1f},
    { 1.0f,  0.4f, -1.1f},
    { 1.0f, -0.4f, -1.1f},
    {-1.0f, -0.4f, -1.1f}
},
/* Left border */
{
    {-1.1f,  0.4f, 1.0f},
    {-1.0f,  0.4f, 1.0f},
    {-1.0f, -0.4f, 1.0f},
    {-1.1f, -0.4f, 1.0f},

    {-1.1f,  0.4f, -1.0f},
    {-1.0f,  0.4f, -1.0f},
    {-1.0f, -0.4f, -1.0f},
    {-1.1f, -0.4f, -1.0f},
},
/* Bottom border */
{
    {-1.0f, -0.4f, 1.0f},
    { 1.0f, -0.4f, 1.0f},
    { 1.0f, -0.9f, 1.0f},
    {-1.0f, -0.9f, 1.0f},

    {-1.0f, -0.4f, -1.0f},
    { 1.0f, -0.4f, -1.0f},
    { 1.0f, -0.9f, -1.0f},
    {-1.0f, -0.9f, -1.0f},
},
/* Top border */
{
    {-1.0f,  0.9f,  1.0f},
    { 1.0f,  0.9f,  1.0f},
    { 1.0f,  0.4f,  1.0f},
    {-1.0f,  0.4f,  1.0f},

    {-1.0f,  0.9f, -1.0f},
    { 1.0f,  0.9f, -1.0f},
    { 1.0f,  0.4f, -1.0f},
    {-1.0f,  0.4f, -1.0f},
}

};

/* Fill vertex buffer */
void CreateVertexBuffer() {
    glGenBuffers(6, vertexbuffer);
    for (auto face = 0; face < 6; face++) {
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer[face]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(g_buffer_data[face]), g_buffer_data[face], GL_STATIC_DRAW);
    }
}

/* Fill texture buffer */
void CreateTextureBuffer() {
    const char* sources[]= {
            "assets/wall.png",
            "assets/wall.png",
            "assets/wall.png",
            "assets/wall.png",
            "assets/floor.png",
            "assets/sky.png"
    };
    int width, height;
    unsigned char* image;

    glGenTextures(6, texturebuffer);
    for (auto face = 0; face < 6; face++) {
        glActiveTexture(GL_TEXTURE0 + texturebuffer[face]);
        glBindTexture(GL_TEXTURE_2D, texturebuffer[face]);

        image = SOIL_load_image(sources[face], &width, &height, 0, SOIL_LOAD_RGB);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
        SOIL_free_image_data(image);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    }
}

}//namespace


/* Constructors */
Wall::Wall()
     :Wall(0, 1.0f, glm::vec4(glm::vec3(0.0f), 1.0f)) {}

Wall::Wall(int face, float scale, glm::vec4 location)
{
    glm::vec4 move = glm::vec4(0.0f);

    if (vertexbuffer[0] == 0) {
        CreateVertexBuffer();
        CreateTextureBuffer();
    }

    face_ = face;
    type_ = kMODEL3D_WALL;
    border_ = engine::geometry::Box(scale, location, move, coords[face_]);

    auto quad_ptr{std::make_unique<engine::geometry::Quad>(face_, scale, location, move,vertexbuffer[face_], texturebuffer[face_])};
    elements_.push_back(std::move(quad_ptr));
}

}//namespace geometry
}//namespace engine
