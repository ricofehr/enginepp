/**
 *  @file shader.h
 *  @brief Shader abstract interface class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_RENDERER_SHADER_H_
#define NEXTFLOOR_RENDERER_SHADER_H_


namespace nextfloor {

namespace renderer {

/**
 *  Shader Interface class, define design for all program shaders
 */
class Shader {

public:
    virtual ~Shader() = default;

    virtual void LoadShader() = 0;

    virtual void LinkShader() = 0;
    virtual void DetachShader() = 0;
    virtual void CheckShader() = 0;
    virtual void CheckProgram() = 0;

    virtual unsigned shader_id() = 0;
};

}  // namespace renderer

}  // namespace nextfloor

#endif  // NEXTFLOOR_RENDERER_SHADER_H_