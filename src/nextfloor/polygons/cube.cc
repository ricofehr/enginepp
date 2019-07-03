/**
 *  @file cube.cc
 *  @brief Cube class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/polygons/cube.h"

#include "nextfloor/renderer/game_window.h"

#include <iostream>

namespace nextfloor {

namespace polygons {

namespace {

static GLuint sElementBuffer = 0;

/* Load element coordinates into buffer */
static void CreateElementBuffer() {
    GLuint elements[] = {
        /* front */
		0, 1, 2,
		2, 3, 0,
		/* right */
		4, 5, 6,
		6, 7, 4,
		/* back */
		8, 9, 10,
		10, 11, 8,
		/* left */
		12, 13, 14,
		14, 15, 12,
		/* bottom */
		16, 17, 18,
		18, 19, 16,
		/* top */
		20, 21, 22,
		22, 23, 20,
    };

    glGenBuffers(1, &sElementBuffer);
    assert(sElementBuffer != 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sElementBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_STREAM_DRAW);
}

} // anonymous namespace

Cube::Cube(glm::vec3 scale, glm::vec4 location)
     :Cube(scale, location, 0, 0) {}

Cube::Cube(float scale, glm::vec4 location,
           GLuint vertexbuffer, GLuint texturebuffer)
    :Cube(glm::vec3(scale), location, vertexbuffer, texturebuffer) {}

Cube::Cube(glm::vec3 scale, glm::vec4 location,
           GLuint vertexbuffer, GLuint texturebuffer)
{
    location_ = location;
    scale_ = scale;
    vertexbuffer_ = vertexbuffer;
    texturebuffer_ = texturebuffer;

    if (sElementBuffer == 0) {
        CreateElementBuffer();
    }
}

void Cube::Draw() noexcept
{
    using nextfloor::renderer::GameWindow;

    if (vertexbuffer_ == 0) {
        return;
    }

    glEnable(GL_CULL_FACE);
    glUniformMatrix4fv(GameWindow::getMatrixId(), 1, GL_FALSE, &mvp_[0][0]);

    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer_);

    /* 1st attribute buffer : vertices position */
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                          8 * sizeof(GLfloat), (void*)0);

    /* 2nd attribute buffer : colors */
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
                          8 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));

    /* 3th attribute buffer : texture position */
    glEnableVertexAttribArray(2);
    glActiveTexture(GL_TEXTURE0 + texturebuffer_);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE,
                          8 * sizeof(GLfloat), (void*)(6 * sizeof(GLfloat)));

    glUniform1i(glGetUniformLocation(GameWindow::getProgramId(), "tex"), texturebuffer_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sElementBuffer);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
}

} // namespace graphics

} // namespace nextfloor
