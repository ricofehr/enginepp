/**
 *   Brick class file
 *   @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "engine/universe/brick.h"

#include <GL/glew.h>
#include <SOIL/SOIL.h>

#include <iostream>
#include <memory>

#include "engine/graphics/border.h"
#include "engine/graphics/cube.h"

namespace engine {

namespace universe {

namespace {

/* GL Buffer ids for texture and vertex */
static GLuint sTextureBuffer = 0;
static GLuint sVertexBuffer = 0;

/* Brick vertex (3) / color (3) / texture (2) coordinates */
const GLfloat sBufferData[192] = {
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

/*
 *  Fill vertex buffer 
 */
void CreateVertexBuffer() noexcept {
    glGenBuffers(1, &sVertexBuffer);
    assert(sVertexBuffer != 0);

    glBindBuffer(GL_ARRAY_BUFFER, sVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(sBufferData), sBufferData, GL_STREAM_DRAW);
}

/*
 *  Fill texture buffer
 */
void CreateTextureBuffer() noexcept {
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

} // anonymous namespace

Brick::Brick()
      :Brick(1.0f, glm::vec4(glm::vec3(0.0f), 1.0f), glm::vec4(0.0f)) {}

Brick::Brick(float scale, glm::vec4 location)
      :Brick(scale, location, glm::vec4(0.0f)) {}

Brick::Brick(float scale, glm::vec4 location, glm::vec4 move)
{
    using engine::graphics::Border;
    using engine::graphics::Cube;

    /* Must be created before with static CreateBuffers function */
    if (sVertexBuffer == 0) {
        CreateVertexBuffer();
        CreateTextureBuffer();
    }

    type_ = kMODEL3D_BRICK;
    border_ = std::make_unique<Border>(scale, location, move);
    auto cube_ptr {std::make_unique<Cube>(scale, location, move,
                                          sVertexBuffer, sTextureBuffer)};
    elements_.push_back(std::move(cube_ptr));
}

void Brick::CreateBuffers() noexcept
{
    if (sVertexBuffer == 0) {
        CreateVertexBuffer();
        CreateTextureBuffer();
    }
}

} // namespace graphics

} // namespace engine
