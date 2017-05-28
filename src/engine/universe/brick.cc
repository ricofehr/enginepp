/*
* Brick class file
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#include "engine/universe/brick.h"

#include <GL/glew.h>
#include <SOIL/SOIL.h>

#include <iostream>
#include <memory>

#include "engine/geometry/box.h"
#include "engine/geometry/cube.h"

namespace engine {
namespace universe {

namespace {

static GLuint sTextureBuffer = 0;
static GLuint sVertexBuffer = 0;

const GLfloat sBufferData[192] = {
  /* Position            Color              Texcoords */
    /* Front */
    -1.0f, -1.0f,  1.0f,  1.0f, 1.0f, 1.0f,  0.0f,  0.0f,
     1.0f, -1.0f,  1.0f,  1.0f, 1.0f, 1.0f,  1.0f,  0.0f,
     1.0f,  1.0f,  1.0f,  1.0f, 1.0f, 1.0f,  1.0f,  1.0f,
    -1.0f,  1.0f,  1.0f,  1.0f, 1.0f, 1.0f,  0.0f,  1.0f,
    /* Right */
     1.0f, -1.0f,  1.0f,  1.0f, 1.0f, 1.0f,  0.0f,  0.0f,
     1.0f, -1.0f, -1.0f,  1.0f, 1.0f, 1.0f,  1.0f,  0.0f,
     1.0f,  1.0f, -1.0f,  1.0f, 1.0f, 1.0f,  1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,  1.0f, 1.0f, 1.0f,  0.0f,  1.0f,
    /* Back */
    -1.0f,  1.0f, -1.0f,  1.0f, 1.0f, 1.0f,  0.0f,  0.0f,
     1.0f,  1.0f, -1.0f,  1.0f, 1.0f, 1.0f,  1.0f,  0.0f,
     1.0f, -1.0f, -1.0f,  1.0f, 1.0f, 1.0f,  1.0f,  1.0f,
    -1.0f, -1.0f, -1.0f,  1.0f, 1.0f, 1.0f,  0.0f,  1.0f,
    /* Left */
    -1.0f, -1.0f, -1.0f,  1.0f, 1.0f, 1.0f,  0.0f,  0.0f,
    -1.0f, -1.0f,  1.0f,  1.0f, 1.0f, 1.0f,  1.0f,  0.0f,
    -1.0f,  1.0f,  1.0f,  1.0f, 1.0f, 1.0f,  1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,  1.0f, 1.0f, 1.0f,  0.0f,  1.0f,
    /* Bottom */
    -1.0f, -1.0f, -1.0f,  1.0f, 1.0f, 1.0f,  0.0f,  0.0f,
     1.0f, -1.0f, -1.0f,  1.0f, 1.0f, 1.0f,  1.0f,  0.0f,
     1.0f, -1.0f,  1.0f,  1.0f, 1.0f, 1.0f,  1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f,  1.0f, 1.0f, 1.0f,  0.0f,  1.0f,
    /* Top */
    -1.0f,  1.0f,  1.0f,  1.0f, 1.0f, 1.0f,  0.0f,  0.0f,
     1.0f,  1.0f,  1.0f,  1.0f, 1.0f, 1.0f,  1.0f,  0.0f,
     1.0f,  1.0f, -1.0f,  1.0f, 1.0f, 1.0f,  1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,  1.0f, 1.0f, 1.0f,  0.0f,  1.0f,
};

/* Fill vertex buffer */
void CreateVertexBuffer() {
    glGenBuffers(1, &sVertexBuffer);
    assert(sVertexBuffer != 0);

    glBindBuffer(GL_ARRAY_BUFFER, sVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(sBufferData), sBufferData, GL_STREAM_DRAW);
}

/* Fill texture buffer */
void CreateTextureBuffer() {
    int width, height;
    unsigned char* image;

    glGenTextures(1, &sTextureBuffer);
    assert(sTextureBuffer != 0);

    glActiveTexture(GL_TEXTURE0 + sTextureBuffer);
    glBindTexture(GL_TEXTURE_2D, sTextureBuffer);

    image = SOIL_load_image("assets/brique.png", &width, &height, 0, SOIL_LOAD_RGB);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
    SOIL_free_image_data(image);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
}

}//namespace

/* Constructors */
Brick::Brick()
      :Brick(1.0f, glm::vec4(glm::vec3(0.0f), 1.0f), glm::vec4(0.0f)) {}

Brick::Brick(float scale, glm::vec4 location)
      :Brick(scale, location, glm::vec4(0.0f)) {}

Brick::Brick(float scale, glm::vec4 location, glm::vec4 move)
{
    using engine::geometry::Box;
    using engine::geometry::Cube;

    /* Must be created before with static CreateBuffers function */
    if (sVertexBuffer == 0) {
        CreateVertexBuffer();
        CreateTextureBuffer();
    }

    type_ = kMODEL3D_BRICK;
    border_ = Box(scale, location, move);
    auto cube_ptr {std::make_unique<Cube>(scale, location, move,
                                          sVertexBuffer, sTextureBuffer)};
    elements_.push_back(std::move(cube_ptr));
}

/* Create global vertex and texture buffers */
void Brick::CreateBuffers()
{
    if (sVertexBuffer == 0) {
        CreateVertexBuffer();
        CreateTextureBuffer();
    }
}

}//namespace geometry
}//namespace engine
