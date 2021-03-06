/**
 *  @file gl_cube_renderer.cc
 *  @brief GlCubeRenderer class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/renderer/cube_gl_renderer_engine.h"

#include <vector>
#include <GL/glew.h>
#include <SOIL/SOIL.h>

namespace nextfloor {

namespace renderer {

namespace {

// clang-format off
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
// clang-format on

}  // namespace

CubeGlRendererEngine::CubeGlRendererEngine(const std::string& texture, GLuint program_id, GLuint matrix_id)
      : GlRendererEngine(texture, program_id, matrix_id)
{}

void CubeGlRendererEngine::Init()
{
    CreateVertexBuffer();
    CreateElementBuffer();
    CreateTextureBuffer();
    is_initialized_ = true;
}

/*
 *  Fill vertex buffer
 */
void CubeGlRendererEngine::CreateVertexBuffer()
{
    glGenBuffers(1, &vertexbuffer_);
    assert(vertexbuffer_ != 0);

    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(sBufferData), sBufferData, GL_STREAM_DRAW);
}

/* Load element coordinates into buffer */
void CubeGlRendererEngine::CreateElementBuffer()
{
    // clang-format off
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
    // clang-format on

    glGenBuffers(1, &elementbuffer_);
    assert(elementbuffer_ != 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_STREAM_DRAW);
}

/*
 *  Fill texture buffer
 */
void CubeGlRendererEngine::CreateTextureBuffer()
{
    int width, height;
    unsigned char* image;

    glGenTextures(1, &texturebuffer_);
    assert(texturebuffer_ != 0);

    glActiveTexture(GL_TEXTURE0 + texturebuffer_);
    glBindTexture(GL_TEXTURE_2D, texturebuffer_);

    /* Load Texture */
    image = SOIL_load_image(texture_.c_str(), &width, &height, 0, SOIL_LOAD_RGB);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
    SOIL_free_image_data(image);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    /* Magnification filter (Stretch the texture) */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    /* Minification Filter (Shrink the texture) */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    /* Create mipmap images */
    glGenerateMipmap(GL_TEXTURE_2D);
}

void CubeGlRendererEngine::Draw(const glm::mat4& mvp)
{
    {
        if (!is_initialized_) {
            Init();
        }

        glEnable(GL_CULL_FACE);
        /* Assign projection matrix to drawn */
        glUniformMatrix4fv(matrix_id_, 1, GL_FALSE, &mvp[0][0]);

        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer_);

        /* 1st attribute buffer : vertices position, used 3 floats */
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (void*)0);

        /* 2nd attribute buffer : colors, used 3 floats */
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));

        /* 3th attribute buffer : texture position, used 2 floats */
        glEnableVertexAttribArray(2);
        glActiveTexture(GL_TEXTURE0 + texturebuffer_);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (void*)(6 * sizeof(GLfloat)));

        /* Assign texture to drawn */
        glUniform1i(glGetUniformLocation(program_id_, "tex"), texturebuffer_);

        /* Draw triangles from vertices setted from mvp matrix thanks elements coordinates */
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementbuffer_);
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);

        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        glDisableVertexAttribArray(2);
    }
}


}  // namespace renderer

}  // namespace nextfloor