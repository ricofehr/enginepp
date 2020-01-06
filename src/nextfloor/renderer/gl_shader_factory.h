/**
 *  @file gl_shader_factory.h
 *  @brief GlShaderFactory Class for shader flyweights objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_RENDERER_GLSHADERFACTORY_H_
#define NEXTFLOOR_RENDERER_GLSHADERFACTORY_H_

#include "nextfloor/renderer/shader_factory.h"

#include <map>
#include <string>
#include <memory>

#include "nextfloor/renderer/shader.h"

namespace nextfloor {

namespace renderer {

/**
 *  @class GlShaderFactory
 *  @brief GlShaderFactory Pattern for Universe Objects
 */
class GlShaderFactory : public ShaderFactory {

public:
    GlShaderFactory();
    ~GlShaderFactory() noexcept final;

    GlShaderFactory(GlShaderFactory&&) = default;
    GlShaderFactory& operator=(GlShaderFactory&&) = default;
    GlShaderFactory(const GlShaderFactory&) = delete;
    GlShaderFactory& operator=(const GlShaderFactory&) = delete;

    Shader* MakeVertexShader(const std::string& shader_path, unsigned int program_id) final;
    Shader* MakeFragmentShader(const std::string& shader_path, unsigned int program_id) final;

private:
    std::map<std::string, std::unique_ptr<Shader>> shaders_;
};

}  // namespace renderer

}  // namespace nextfloor

#endif  // NEXTFLOOR_RENDERER_GLSHADERFACTORY_H_
