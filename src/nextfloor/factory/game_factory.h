/**
 *  @file game_factory.h
 *  @brief GameFactory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_FACTORY_GAMEFACTORY_H_
#define NEXTFLOOR_FACTORY_GAMEFACTORY_H_

#include "nextfloor/factory/factory.h"

#include <memory>
#include <string>
#include <glm/glm.hpp>

#include "nextfloor/factory/renderer_factory.h"

namespace nextfloor {

namespace factory {

/**
 *  @class GameFactory
 *  @brief GameFactory Pattern for Universe Objects
 */
class GameFactory : public Factory {

public:

    GameFactory();

    GameFactory(GameFactory&&) = default;
    GameFactory& operator=(GameFactory&&) = default;
    GameFactory(const GameFactory&) = default;
    GameFactory& operator=(const GameFactory&) = default;

    virtual ~GameFactory() override;

    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeUniverse() const noexcept override;
    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeRoom(glm::vec3 location) const noexcept override;

    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeFrontWall(glm::vec3 location, glm::vec3 scale) const noexcept override;
    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeRightWall(glm::vec3 location, glm::vec3 scale) const noexcept override;
    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeBackWall(glm::vec3 location, glm::vec3 scale) const noexcept override;
    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeLeftWall(glm::vec3 location, glm::vec3 scale) const noexcept override;
    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeFloor(glm::vec3 location, glm::vec3 scale) const noexcept override;
    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeRoof(glm::vec3 location, glm::vec3 scale) const noexcept override;
    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeWallBrick(glm::vec3 location, glm::vec3 scale, std::string texture) const noexcept override;

    virtual std::unique_ptr<nextfloor::objects::Mesh> MakePlayer(glm::vec3 location) const noexcept override;

    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeRock(glm::vec3 location) const noexcept override;
    virtual std::unique_ptr<nextfloor::objects::Mesh> MakeLittleRock(glm::vec3 location) const noexcept override;

    virtual std::unique_ptr<nextfloor::objects::Camera> MakeCamera(nextfloor::objects::Mesh* owner) const noexcept override;

    virtual std::unique_ptr<nextfloor::objects::Polygon> MakeCube(glm::vec3 location, glm::vec3 scale) const noexcept override;

    virtual std::unique_ptr<nextfloor::objects::Border> MakeBorder(glm::vec3 location, glm::vec3 scale) const noexcept override;

    virtual std::unique_ptr<nextfloor::objects::Grid> MakeUniverseGrid(nextfloor::objects::Mesh* universe) const noexcept override;
    virtual std::unique_ptr<nextfloor::objects::Grid> MakeRoomGrid(nextfloor::objects::Mesh* room) const noexcept override;
    virtual std::unique_ptr<nextfloor::objects::Grid> MakeGrid(nextfloor::objects::Mesh* owner, glm::ivec3 boxes_count, glm::vec3 box_dimension) const noexcept override;
    virtual std::unique_ptr<nextfloor::objects::GridBox> MakeRoomGridBox(glm::vec3 grid_coords, nextfloor::objects::Grid* room_grid) const noexcept override;
    virtual std::unique_ptr<nextfloor::objects::GridBox> MakeUniverseGridBox(glm::vec3 grid_coords, nextfloor::objects::Grid* universe_grid) const noexcept override;
    virtual std::unique_ptr<nextfloor::objects::GridBox> MakeGridBox(glm::vec3 grid_coords, nextfloor::objects::Grid* grid) const noexcept override;

    virtual nextfloor::objects::RendererEngine* MakeCubeRenderer(std::string texture) const noexcept override;
    virtual std::unique_ptr<nextfloor::renderer::Shader> MakeVertexShader(std::string shader_path) const noexcept override;
    virtual std::unique_ptr<nextfloor::renderer::Shader> MakeFragmentShader(std::string shader_path) const noexcept override;

    virtual std::unique_ptr<nextfloor::objects::CollisionEngine> MakeCollisionEngine() const noexcept override;

    virtual std::unique_ptr<nextfloor::gameplay::Level> MakeLevel() const noexcept override;

private:

    std::unique_ptr<RendererFactory> renderer_factory_;
};

} // namespace factory

} // namespace nextfloor

#endif // NEXTFLOOR_FACTORY_GAMEFACTORY_H_
