/**
 *  @file mesh_factory.h
 *  @brief MeshFactory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_MODELMESHFACTORY_H_
#define NEXTFLOOR_OBJECTS_MODELMESHFACTORY_H_

#include "nextfloor/objects/mesh_factory.h"

namespace nextfloor {

namespace objects {

/**
 *  @class MeshFactory
 *  @brief Concret factory for Universe Objects
 */
class ModelMeshFactory : public MeshFactory {

public:
    std::unique_ptr<Mesh> MakeUniverse() const final;
    std::unique_ptr<Mesh> MakeRoom(const glm::vec3& location) const final;

    std::unique_ptr<Mesh> MakeFrontWall(const glm::vec3& location, const glm::vec3& scale) const final;
    std::unique_ptr<Mesh> MakeRightWall(const glm::vec3& location, const glm::vec3& scale) const final;
    std::unique_ptr<Mesh> MakeBackWall(const glm::vec3& location, const glm::vec3& scale) const final;
    std::unique_ptr<Mesh> MakeLeftWall(const glm::vec3& location, const glm::vec3& scale) const final;
    std::unique_ptr<Mesh> MakeFloor(const glm::vec3& location, const glm::vec3& scale) const final;
    std::unique_ptr<Mesh> MakeRoof(const glm::vec3& location, const glm::vec3& scale) const final;
    std::unique_ptr<Mesh> MakeWallBrick(const glm::vec3& location,
                                        const glm::vec3& scale,
                                        const std::string& texture) const final;

    std::unique_ptr<Mesh> MakePlayer(const glm::vec3& location) const final;

    std::unique_ptr<Mesh> MakeRock(const glm::vec3& location) const final;
    std::unique_ptr<Mesh> MakeLittleRock(const glm::vec3& location) const final;

    std::unique_ptr<Camera> MakeCamera(Mesh* owner) const final;

    std::unique_ptr<Polygon> MakeCube(const glm::vec3& location, const glm::vec3& scale) const final;
    std::unique_ptr<Polygon> MakeCube(const glm::vec3& location,
                                      const glm::vec3& scale,
                                      const std::string& texture) const final;

    std::unique_ptr<Border> MakeBorder(const glm::vec3& location, const glm::vec3& scale) const final;

    std::unique_ptr<Grid> MakeUniverseGrid(Mesh* universe) const final;
    std::unique_ptr<Grid> MakeRoomGrid(Mesh* room) const final;
    std::unique_ptr<Grid> MakeGrid(Mesh* owner, const glm::ivec3& boxes_count, const glm::vec3& box_dimension) const final;
    std::unique_ptr<GridBox> MakeRoomGridBox(const glm::vec3& grid_coords, Grid* room_grid) const final;
    std::unique_ptr<GridBox> MakeUniverseGridBox(const glm::vec3& coords, Grid* grid) const final;
    std::unique_ptr<GridBox> MakeGridBox(const glm::vec3& grid_coords, Grid* grid) const final;

    CollisionEngine* MakeCollisionEngine() const final;

    std::unique_ptr<nextfloor::gameplay::Level> MakeLevel() const override;
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_OBJECTS_MODELMESHFACTORY_H_
