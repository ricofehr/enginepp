/**
 *  @file shader_factory.h
 *  @brief Abstract Factory Class for shader objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_RENDERER_SHADERFACTORY_H_
#define NEXTFLOOR_RENDERER_SHADERFACTORY_H_

#include <memory>
#include <glm/glm.hpp>


#include "nextfloor/renderer/shader.h"

namespace nextfloor {

namespace renderer {

/**
 *  @class ShaderFactory
 *  @brief Abstract Factory Pattern for flyweight shader objects
 */
class ShaderFactory {

public:
    virtual ~ShaderFactory() = default;

    virtual Shader* MakeVertexShader(const std::string& shader_path, unsigned int program_id) = 0;
    virtual Shader* MakeFragmentShader(const std::string& shader_path, unsigned int program_id) = 0;
};

}  // namespace renderer

}  // namespace nextfloor

#endif  // NEXTFLOOR_GAMEPLAY_RENDERERFACTORY_H_
