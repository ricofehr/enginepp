/**
 *  @file facade_factory.h
 *  @brief Abstract Factory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_FACTORY_FACADEFACTORY_H_
#define NEXTFLOOR_FACTORY_FACADEFACTORY_H_

#include <memory>
#include <string>
#include <glm/glm.hpp>

#include "nextfloor/objects/mesh.h"
#include "nextfloor/objects/border.h"
#include "nextfloor/objects/collision_engine.h"
#include "nextfloor/objects/grid.h"
#include "nextfloor/objects/grid_box.h"
#include "nextfloor/objects/polygon.h"
#include "nextfloor/objects/camera.h"
#include "nextfloor/renderer/shader.h"
#include "nextfloor/gameplay/level.h"

#include "nextfloor/gameplay/renderer_engine.h"
#include "nextfloor/renderer/shader.h"
#include "nextfloor/gameplay/scene_window.h"
#include "nextfloor/renderer/scene_input.h"

#include "nextfloor/factory/renderer_factory.h"

namespace nextfloor {

namespace factory {

/**
 *  @class FacadeFactory
 *  @brief Abstract Factory Pattern for Universe Objects
 */
class FacadeFactory {

public:
    FacadeFactory();
    ~FacadeFactory() noexcept;

    FacadeFactory(FacadeFactory&&) = default;
    FacadeFactory& operator=(FacadeFactory&&) = default;
    FacadeFactory(const FacadeFactory&) = delete;
    FacadeFactory& operator=(const FacadeFactory&) = delete;

    /* RendererFactory Part */
    nextfloor::gameplay::RendererEngine* MakeCubeRenderer(const std::string& texture);
    nextfloor::renderer::Shader* MakeVertexShader(const std::string& shader_path, unsigned int program_id);
    nextfloor::renderer::Shader* MakeFragmentShader(const std::string& shader_path, unsigned int program_id);
    nextfloor::gameplay::SceneWindow* MakeSceneWindow();
    std::unique_ptr<nextfloor::renderer::SceneInput> MakeSceneInput();

private:
    std::unique_ptr<RendererFactory> renderer_factory_;
};

}  // namespace factory

}  // namespace nextfloor

#endif  // NEXTFLOOR_FACTORY_FACADEFACTORY_H_
