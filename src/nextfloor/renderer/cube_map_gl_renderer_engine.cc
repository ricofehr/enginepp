/**
 *  @file gl_cube_renderer.cc
 *  @brief GlCubeRenderer class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/renderer/cube_map_gl_renderer_engine.h"

#include <vector>
#include <GL/glew.h>
#include <SOIL/SOIL.h>

namespace nextfloor {

namespace renderer {

namespace {

// clang-format off
/* Brick vertex (3) / color (3) / texture (2) coordinates */
const GLfloat sBufferData[108] = {
    -1.0f,  1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f,
    1.0f, -1.0f, -1.0f,
    1.0f, -1.0f, -1.0f,
    1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f, -1.0f,

    -1.0f, -1.0f,  1.0f,
    -1.0f, -1.0f, -1.0f,
    -1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f,

    1.0f, -1.0f, -1.0f,
    1.0f, -1.0f,  1.0f,
    1.0f,  1.0f,  1.0f,
    1.0f,  1.0f,  1.0f,
    1.0f,  1.0f, -1.0f,
    1.0f, -1.0f, -1.0f,

    -1.0f, -1.0f,  1.0f,
    -1.0f,  1.0f,  1.0f,
    1.0f,  1.0f,  1.0f,
    1.0f,  1.0f,  1.0f,
    1.0f, -1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f,

    -1.0f,  1.0f, -1.0f,
    1.0f,  1.0f, -1.0f,
    1.0f,  1.0f,  1.0f,
    1.0f,  1.0f,  1.0f,
    -1.0f,  1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,

    -1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f,  1.0f,
    1.0f, -1.0f, -1.0f,
    1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f,  1.0f,
    1.0f, -1.0f,  1.0f
};
// clang-format on

}  // namespace

CubeMapGlRendererEngine::CubeMapGlRendererEngine(PipelineProgram* pipeline_program_)
      : GlRendererEngine(pipeline_program_)
{}

void CubeMapGlRendererEngine::Init()
{
    CreateVertexBuffer();
    CreateTextureBuffer();
    is_initialized_ = true;
}

/*
 *  Fill vertex buffer
 */
void CubeMapGlRendererEngine::CreateVertexBuffer()
{
    glGenVertexArrays(1, &vertexarray_);
    glGenBuffers(1, &vertexbuffer_);
    assert(vertexbuffer_ != 0);

    glBindVertexArray(vertexarray_);
    glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(sBufferData), sBufferData, GL_STREAM_DRAW);
}

/*
 *  Fill texture buffer
 */
void CubeMapGlRendererEngine::CreateTextureBuffer()
{
    int width, height;
    std::string textures_faces[6] = {"assets/cubemap/right.png",
                                     "assets/cubemap/left.png",
                                     "assets/cubemap/top.png",
                                     "assets/cubemap/bottom.png",
                                     "assets/cubemap/front.png",
                                     "assets/cubemap/back.png"};

    glGenTextures(1, &texturebuffer_);
    assert(texturebuffer_ != 0);

    glBindTexture(GL_TEXTURE_CUBE_MAP, texturebuffer_);

    for(GLuint i = 0; i < 6; i++) {
        /* Load Texture */
        unsigned char* image = SOIL_load_image(textures_faces[i].c_str(), &width, &height, 0, SOIL_LOAD_RGB);
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
        SOIL_free_image_data(image);
    }

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_LEVEL, 0);
}

void CubeMapGlRendererEngine::Draw(const glm::mat4& mvp)
{
    {
        if (!is_initialized_) {
            Init();
        }

        glDepthFunc(GL_LEQUAL);

        glUseProgram(pipeline_program_->getProgramId());

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        /* Assign projection matrix to drawn */
        glUniformMatrix4fv(pipeline_program_->getMatrixId(), 1, GL_FALSE, &mvp[0][0]);
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer_);

        glBindVertexArray(vertexarray_);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, texturebuffer_);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glBindVertexArray(0);
        glDisableVertexAttribArray(0);
        glDepthFunc(GL_LESS);
    }
}


}  // namespace renderer

}  // namespace nextfloor
