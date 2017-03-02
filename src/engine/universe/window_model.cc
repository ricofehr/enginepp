/*
* WindowModel class file
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#include "engine/universe/window_model.h"

#include <glm/glm.hpp>
#include <SOIL/SOIL.h>

#include <iostream>
#include <memory>

namespace engine {
namespace universe {

namespace {
GLuint vertexbuffer[4] = {0,0,0,0};
GLuint texturebuffer[4] = {0,0,0,0};

const GLfloat g_buffer_data[4][32] = {
/*   Position             Color              Texcoords */
{   /* Front */
    -0.2f, -0.1f,  0.0f,  1.0f, 1.0f, 1.0f,  0.0f,  0.0f,
     0.2f, -0.1f,  0.0f,  1.0f, 1.0f, 1.0f,  1.0f,  0.0f,
     0.2f,  0.1f,  0.0f,  1.0f, 1.0f, 1.0f,  1.0f,  1.0f,
    -0.2f,  0.1f,  0.0f,  1.0f, 1.0f, 1.0f,  0.0f,  1.0f,
},

{
     /* Right */
     0.0f, -0.1f,  0.2f,  1.0f, 1.0f, 1.0f,  0.0f,  0.0f,
     0.0f, -0.1f, -0.2f,  1.0f, 1.0f, 1.0f,  1.0f,  0.0f,
     0.0f,  0.1f, -0.2f,  1.0f, 1.0f, 1.0f,  1.0f,  1.0f,
     0.0f,  0.1f,  0.2f,  1.0f, 1.0f, 1.0f,  0.0f,  1.0f,
},

{   /* Back */
    -0.2f,  0.1f, 0.0f,  1.0f, 1.0f, 1.0f,  0.0f,  1.0f,
     0.2f,  0.1f, 0.0f,  1.0f, 1.0f, 1.0f,  1.0f,  1.0f,
     0.2f, -0.1f, 0.0f,  1.0f, 1.0f, 1.0f,  1.0f,  0.0f,
    -0.2f, -0.1f, 0.0f,  1.0f, 1.0f, 1.0f,  0.0f,  0.0f,
},

{   /* Left */
    -0.0f, -0.1f, -0.2f,  1.0f, 1.0f, 1.0f,  0.0f,  0.0f,
    -0.0f, -0.1f,  0.2f,  1.0f, 1.0f, 1.0f,  1.0f,  0.0f,
    -0.0f,  0.1f,  0.2f,  1.0f, 1.0f, 1.0f,  1.0f,  1.0f,
    -0.0f,  0.1f, -0.2f,  1.0f, 1.0f, 1.0f,  0.0f,  1.0f,
},

};


const std::vector<glm::vec3> coords[4] = {
/* Front Border */
{
    {-0.2f,  0.1f,  0.1f},
    { 0.2f,  0.1f,  0.1f},
    { 0.2f, -0.1f,  0.1f},
    {-0.2f, -0.1f,  0.1f},

    {-0.2f,  0.1f,  0.0f},
    { 0.2f,  0.1f,  0.0f},
    { 0.2f, -0.1f,  0.0f},
    {-0.2f, -0.1f,  0.0f},
},
/* Right border */
{
    {0.0f,  0.1f, 0.1f},
    {0.1f,  0.1f, 0.1f},
    {0.1f, -0.1f, 0.1f},
    {0.0f, -0.1f, 0.1f},

    {0.0f,  0.1f, -0.1f},
    {0.1f,  0.1f, -0.1f},
    {0.1f, -0.1f, -0.1f},
    {0.0f, -0.1f, -0.1f},
},
/* Back border */
{
    {-0.1f,  0.1f, 0.0f},
    { 0.1f,  0.1f, 0.0f},
    { 0.1f, -0.1f, 0.0f},
    {-0.1f, -0.1f, 0.0f},

    {-0.1f,  0.1f, -0.1f},
    { 0.1f,  0.1f, -0.1f},
    { 0.1f, -0.1f, -0.1f},
    {-0.1f, -0.1f, -0.1f}
},
/* Left border */
{
    {0.0f,  0.1f, 0.1f},
    {0.1f,  0.1f, 0.1f},
    {0.1f, -0.1f, 0.1f},
    {0.0f, -0.1f, 0.1f},

    {0.0f,  0.1f, -0.1f},
    {0.1f,  0.1f, -0.1f},
    {0.1f, -0.1f, -0.1f},
    {0.0f, -0.1f, -0.1f},
}
};

/* Fill vertex buffer */
void CreateVertexBuffer() {
    glGenBuffers(4, vertexbuffer);
    for (auto face = 0; face < 4; face++) {
        // The following commands will talk about our 'vertexbuffer' buffer
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer[face]);
        // Give our vertices to OpenGL.
        glBufferData(GL_ARRAY_BUFFER, sizeof(g_buffer_data[face]), g_buffer_data[face], GL_STATIC_DRAW);
    }
}

/* Fill texture buffer */
void CreateTextureBuffer() {
    int width, height;
    unsigned char* image;

    glGenTextures(4, texturebuffer);
    for (auto face = 0; face < 4; face++) {
        glActiveTexture(GL_TEXTURE0 + texturebuffer[face]);
        glBindTexture(GL_TEXTURE_2D, texturebuffer[face]);

        image = SOIL_load_image("assets/window_model.png", &width, &height, 0, SOIL_LOAD_RGB);
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
WindowModel::WindowModel()
     :WindowModel(0, 1.0f, glm::vec4(glm::vec3(0.0f), 1.0f)) {}

WindowModel::WindowModel(int face, float scale, glm::vec4 location)
{
    glm::vec4 move = glm::vec4(0.0f);

    if (vertexbuffer[0] == 0) {
        CreateVertexBuffer();
        CreateTextureBuffer();
    }

    face_ = face;
    type_ = kMODEL3D_DOOR;
    border_ = engine::geometry::Box(scale, location, move, coords[face_]);

    auto quad_ptr{std::make_unique<engine::geometry::Quad>(face_, scale, location, move,vertexbuffer[face_], texturebuffer[face_])};
    elements_.push_back(std::move(quad_ptr));
}

}//namespace geometry
}//namespace engine
