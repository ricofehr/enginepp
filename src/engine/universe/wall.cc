/*
* Wall class file
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#include "engine/universe/wall.h"

#include <GL/glew.h>
#include <SOIL/SOIL.h>

#include <iostream>
#include <memory>

#include "engine/geometry/box.h"
#include "engine/geometry/cube.h"

namespace engine {
namespace universe {

namespace {
GLuint texturebuffer[3] = {0,0,0};
GLuint vertexbuffer[3] = {0,0,0};

const GLfloat g_buffer_data[192] = {
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
    glGenBuffers(3, vertexbuffer);

    GLfloat buffer_data[192];
    for (auto i = 0; i < 192; i++)
        buffer_data[i] = g_buffer_data[i];

    for (auto face = 0; face < 3; face++) {
        /*
        if (face) {
            for (auto i = 6; i < 192; i+=8) {
                buffer_data[i] = g_buffer_data[i] * face * 4.0f;
                buffer_data[i+1] = g_buffer_data[i+1] * face * 4.0f;
            }
        }
        */
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer[face]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(buffer_data), buffer_data, GL_STATIC_DRAW);
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

    glGenTextures(3, texturebuffer);
    for (auto face = 0; face < 3; face++) {
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
      :Wall(glm::vec3(1.0f), glm::vec4(glm::vec3(0.0f), 1.0f), 1) {}

Wall::Wall(glm::vec3 scale, glm::vec4 location, int face)
{
    using engine::geometry::Box;
    using engine::geometry::Cube;

    /* No thread safe execution */
    if (vertexbuffer[0] == 0) {
        CreateVertexBuffer();
        CreateTextureBuffer();
    }

    type_ = kMODEL3D_WALL;
    border_ = Box(scale, location);
    auto cube_ptr {std::make_unique<Cube>(scale, location, glm::vec4(0.0f),
                                          vertexbuffer[face], texturebuffer[face])};
    elements_.push_back(std::move(cube_ptr));
}

void Wall::CreateBuffers()
{
    if (vertexbuffer[0] == 0) {
        CreateVertexBuffer();
        CreateTextureBuffer();
    }
}

}//namespace geometry
}//namespace engine
