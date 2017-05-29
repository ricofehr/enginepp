/*
* Quad class file
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#include "engine/geometry/quad.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <iostream>

#include "engine/renderer/loopgl.h"

namespace engine {

namespace geometry {

namespace {

static GLuint sElementBuffer = 0;

static void CreateElementBuffer() {
    GLuint elements[] = {
        0, 1, 2,
        2, 3, 0
    };

    glGenBuffers(1, &sElementBuffer);
    assert(sElementBuffer != 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sElementBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_STATIC_DRAW);
}

}//namespace

/* Constructors */
Quad::Quad(int face, float scale, glm::vec4 location,
           GLuint vertexbuffer, GLuint texturebuffer)
     :Quad(face, scale, location, glm::vec4(0.0f), vertexbuffer, texturebuffer) {}

Quad::Quad(int face, float scale, glm::vec4 location,
           glm::vec4 move, GLuint vertexbuffer, GLuint texturebuffer)
     :Quad(face, glm::vec3(scale), location, move, vertexbuffer, texturebuffer) {}

Quad::Quad(int face, glm::vec3 scale, glm::vec4 location,
           GLuint vertexbuffer, GLuint texturebuffer)
     :Quad(face, scale, location, glm::vec4(0.0f), vertexbuffer, texturebuffer) {}

Quad::Quad(int face, glm::vec3 scale, glm::vec4 location,
           glm::vec4 move, GLuint vertexbuffer, GLuint texturebuffer)
{
    face_ = face;
    location_ = location;
    scale_ = scale;
    move_ = move;
    vertexbuffer_ = vertexbuffer;
    texturebuffer_ = texturebuffer;
    distance_ = -1;

    if (sElementBuffer == 0) {
        CreateElementBuffer();
    }
}

/* Draw the Quad on the scene */
void Quad::Draw()
{
    using engine::renderer::LoopGL;

    if (vertexbuffer_ == 0) {
        return;
    }

    /* lock for ensure only one object draw buffer in same time */
    glDisable(GL_CULL_FACE);
    glUniformMatrix4fv(LoopGL::sMatrixId, 1, GL_FALSE, &mvp_[0][0]);
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer_);

    /* 3 attributes: vertex, color, and textures */
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (void*)0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
                          8 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));

    glEnableVertexAttribArray(2);
    glActiveTexture(GL_TEXTURE0 + texturebuffer_);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE,
                          8 * sizeof(GLfloat), (void*)(6 * sizeof(GLfloat)));

    glUniform1i(glGetUniformLocation(LoopGL::sProgramId, "tex"), texturebuffer_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sElementBuffer);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
}

}//namespace geometry
}//namespace engine
