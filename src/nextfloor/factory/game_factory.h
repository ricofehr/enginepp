/**
 *  @file game_factory.h
 *  @brief GameFactory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_FACTORY_GAMEFACTORY_H_
#define NEXTFLOOR_FACTORY_GAMEFACTORY_H_

#include "nextfloor/factory/mesh_factory.h"

#include "nextfloor/factory/renderer_factory.h"

namespace nextfloor {

namespace factory {

/**
 *  @class GameFactory
 *  @brief GameFactory Pattern for Universe Objects
 */
class GameFactory : public MeshFactory {

public:
    std::unique_ptr<nextfloor::objects::Mesh> MakeUniverse() const final;
    std::unique_ptr<nextfloor::objects::Mesh> MakeRoom(const glm::vec3& location) const final;

    std::unique_ptr<nextfloor::objects::Mesh> MakeFrontWall(const glm::vec3& location, const glm::vec3& scale) const final;
    std::unique_ptr<nextfloor::objects::Mesh> MakeRightWall(const glm::vec3& location, const glm::vec3& scale) const final;
    std::unique_ptr<nextfloor::objects::Mesh> MakeBackWall(const glm::vec3& location, const glm::vec3& scale) const final;
    std::unique_ptr<nextfloor::objects::Mesh> MakeLeftWall(const glm::vec3& location, const glm::vec3& scale) const final;
    std::unique_ptr<nextfloor::objects::Mesh> MakeFloor(const glm::vec3& location, const glm::vec3& scale) const final;
    std::unique_ptr<nextfloor::objects::Mesh> MakeRoof(const glm::vec3& location, const glm::vec3& scale) const final;
    std::unique_ptr<nextfloor::objects::Mesh> MakeWallBrick(const glm::vec3& location,
                                                            const glm::vec3& scale,
                                                            const std::string& texture) const final;

    std::unique_ptr<nextfloor::objects::Mesh> MakePlayer(const glm::vec3& location) const final;

    std::unique_ptr<nextfloor::objects::Mesh> MakeRock(const glm::vec3& location) const final;
    std::unique_ptr<nextfloor::objects::Mesh> MakeLittleRock(const glm::vec3& location) const final;

    std::unique_ptr<nextfloor::objects::Camera> MakeCamera(nextfloor::objects::Mesh* owner) const final;

    std::unique_ptr<nextfloor::objects::Polygon> MakeCube(const glm::vec3& location, const glm::vec3& scale) const final;
    std::unique_ptr<nextfloor::objects::Polygon> MakeCube(const glm::vec3& location,
                                                          const glm::vec3& scale,
                                                          const std::string& texture) const final;

    std::unique_ptr<nextfloor::objects::Border> MakeBorder(const glm::vec3& location, const glm::vec3& scale) const final;

    std::unique_ptr<nextfloor::objects::Grid> MakeUniverseGrid(nextfloor::objects::Mesh* universe) const final;
    std::unique_ptr<nextfloor::objects::Grid> MakeRoomGrid(nextfloor::objects::Mesh* room) const final;
    std::unique_ptr<nextfloor::objects::Grid> MakeGrid(nextfloor::objects::Mesh* owner,
                                                       const glm::ivec3& boxes_count,
                                                       const glm::vec3& box_dimension) const final;
    std::unique_ptr<nextfloor::objects::GridBox> MakeRoomGridBox(const glm::vec3& grid_coords,
                                                                 nextfloor::objects::Grid* room_grid) const final;
    std::unique_ptr<nextfloor::objects::GridBox> MakeUniverseGridBox(const glm::vec3& coords,
                                                                     nextfloor::objects::Grid* grid) const final;
    std::unique_ptr<nextfloor::objects::GridBox> MakeGridBox(const glm::vec3& grid_coords,
                                                             nextfloor::objects::Grid* grid) const final;

    nextfloor::objects::CollisionEngine* MakeCollisionEngine() const final;

    std::unique_ptr<nextfloor::gameplay::Level> MakeLevel() const override;
};

}  // namespace factory

}  // namespace nextfloor

#endif  // NEXTFLOOR_FACTORY_GAMEFACTORY_H_
