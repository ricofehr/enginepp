/*
* Wall class file
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#include "engine/universe/wall.h"

#include <GL/glew.h>
#include <SOIL/SOIL.h>

#include <iostream>
#include <memory>

#include "engine/graphics/border.h"
#include "engine/graphics/cube.h"

namespace engine {
namespace universe {

namespace {

static GLuint sTextureBuffer[3] = {0,0,0};
static GLuint sVertexBuffer[3] = {0,0,0};

static const GLfloat sBufferData[192] = {
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
    glGenBuffers(3, sVertexBuffer);

    for (auto face = 0; face < 3; face++) {
        glBindBuffer(GL_ARRAY_BUFFER, sVertexBuffer[face]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(sBufferData), sBufferData, GL_STATIC_DRAW);
    }
}

/* Fill texture buffer */
void CreateTextureBuffer() {
    const char* sources[]= {
            "assets/sky.png",
            "assets/wall.png",
            "assets/floor.png"
    };
    int width, height;
    unsigned char* image;

    glGenTextures(3, sTextureBuffer);
    for (auto face = 0; face < 3; face++) {
        glActiveTexture(GL_TEXTURE0 + sTextureBuffer[face]);
        glBindTexture(GL_TEXTURE_2D, sTextureBuffer[face]);

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
      :Wall(glm::vec3(1.0f), glm::vec4(glm::vec3(0.0f), 1.0f), 1) {}

Wall::Wall(glm::vec3 scale, glm::vec4 location, int face)
{
    using engine::graphics::Border;
    using engine::graphics::Cube;

    /* Must be created before with static CreateBuffers function */
    if (sVertexBuffer[0] == 0) {
        CreateVertexBuffer();
        CreateTextureBuffer();
    }

    type_ = kMODEL3D_WALL;
    border_ = Border(scale, location);
    auto cube_ptr {std::make_unique<Cube>(scale, location, glm::vec4(0.0f),
                                          sVertexBuffer[face], sTextureBuffer[face])};
    elements_.push_back(std::move(cube_ptr));
}

/* Create global vertex and texture buffers */
void Wall::CreateBuffers()
{
    if (sVertexBuffer[0] == 0) {
        CreateVertexBuffer();
        CreateTextureBuffer();
    }
}

}//namespace graphics
}//namespace engine
