/*
* Cube class file
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#include "engine/geometry/cube.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <iostream>

#include "engine/helpers/proxygl.h"

namespace engine {

/* Use of global variables defined in proxygl namespace */
namespace helpers {
namespace proxygl {
    extern GLuint kProgramId;
    extern GLuint kMatrixId;
}//namespace proxygl
}//namespace helpers

namespace geometry {

namespace {


static GLuint elementbuffer = 0;

/* Load element coordinates in buffer */
void CreateElementBuffer() {
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

    glGenBuffers(1, &elementbuffer);
    assert(elementbuffer != 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_STREAM_DRAW);
}

}//namespace

/* Constructors */
Cube::Cube(float scale, glm::vec4 location)
     :Cube(scale, location, glm::vec4(0.0f), 0, 0) {}

Cube::Cube(float scale, glm::vec4 location, glm::vec4 move)
     :Cube(scale, location, move, 0, 0) {}

Cube::Cube(glm::vec3 scale, glm::vec4 location)
     :Cube(scale, location, glm::vec4(0.0f), 0, 0) {}

Cube::Cube(glm::vec3 scale, glm::vec4 location, glm::vec4 move)
     :Cube(scale, location, move, 0, 0) {}

Cube::Cube(float scale, glm::vec4 location,
           GLuint vertexbuffer, GLuint texturebuffer)
     :Cube(scale, location, glm::vec4(0.0f), vertexbuffer, texturebuffer) {}

Cube::Cube(float scale, glm::vec4 location, glm::vec4 move,
           GLuint vertexbuffer, GLuint texturebuffer)
    :Cube(glm::vec3(scale), location, move, vertexbuffer, texturebuffer) {}

Cube::Cube(glm::vec3 scale, glm::vec4 location, glm::vec4 move,
           GLuint vertexbuffer, GLuint texturebuffer)
{
    location_ = location;
    scale_ = scale;
    move_ = move;
    vertexbuffer_ = vertexbuffer;
    texturebuffer_ = texturebuffer;
    distance_ = -1;

    if (elementbuffer == 0) {
        CreateElementBuffer();
    }
}

/* Draw the cube */
void Cube::Draw()
{
    /* kProgramId and kMatrixId global fixed values */
    using engine::helpers::proxygl::kMatrixId;
    using engine::helpers::proxygl::kProgramId;

    if (vertexbuffer_ == 0) {
        return;
    }

    glEnable(GL_CULL_FACE);
    glUniformMatrix4fv(kMatrixId, 1, GL_FALSE, &mvp_[0][0]);

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

    glUniform1i(glGetUniformLocation(kProgramId, "tex"), texturebuffer_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
}

}//namespace geometry
}//namespace engine
