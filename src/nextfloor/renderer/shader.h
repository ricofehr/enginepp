/**
 *  @file shader.h
 *  @brief Shader abstract interface class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_RENDERER_SHADER_H_
#define NEXTFLOOR_RENDERER_SHADER_H_

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string>

namespace nextfloor {

namespace renderer {

/**
 *  Shader Interface class, define design for all program shaders
 */
class Shader {

public:
    virtual ~Shader() = default;

    virtual void LoadShader() = 0;

    virtual void LinkShader();
    virtual void DetachShader();
    virtual void CheckShader();
    static void CheckProgram();

    GLuint shader_id() { return shader_id_; }

protected:
    Shader(std::string shader_filepath);

    Shader(Shader&&) = default;
    Shader& operator=(Shader&&) = default;
    Shader(const Shader&) = default;
    Shader& operator=(const Shader&) = default;


    std::string shader_filepath_;
    GLuint shader_id_;
};

}  // namespace renderer

}  // namespace nextfloor

#endif  // NEXTFLOOR_RENDERER_SHADER_H_