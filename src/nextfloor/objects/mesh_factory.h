/**
 *  @file object_factory.h
 *  @brief Abstract Factory Class for game objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_MESHFACTORY_H_
#define NEXTFLOOR_OBJECTS_MESHFACTORY_H_

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

#include "nextfloor/gameplay/renderer_engine.h"
#include "nextfloor/gameplay/level.h"

namespace nextfloor {

namespace objects {

/**
 *  @class MeshFactory
 *  @brief Abstract Factory Pattern for Universe Objects
 */
class MeshFactory {

public:
    virtual ~MeshFactory() = default;

    virtual std::unique_ptr<Mesh> MakeUniverse() const = 0;
    virtual std::unique_ptr<Mesh> MakeRoom(const glm::vec3& location) const = 0;

    virtual std::unique_ptr<Mesh> MakeFrontWall(const glm::vec3& location, const glm::vec3& scale) const = 0;
    virtual std::unique_ptr<Mesh> MakeRightWall(const glm::vec3& location, const glm::vec3& scale) const = 0;
    virtual std::unique_ptr<Mesh> MakeBackWall(const glm::vec3& location, const glm::vec3& scale) const = 0;
    virtual std::unique_ptr<Mesh> MakeLeftWall(const glm::vec3& location, const glm::vec3& scale) const = 0;
    virtual std::unique_ptr<Mesh> MakeFloor(const glm::vec3& location, const glm::vec3& scale) const = 0;
    virtual std::unique_ptr<Mesh> MakeRoof(const glm::vec3& location, const glm::vec3& scale) const = 0;
    virtual std::unique_ptr<Mesh> MakeWallBrick(const glm::vec3& location,
                                                const glm::vec3& scale,
                                                const std::string& texture) const = 0;

    virtual std::unique_ptr<Mesh> MakeRock(const glm::vec3& location) const = 0;
    virtual std::unique_ptr<Mesh> MakeLittleRock(const glm::vec3& location) const = 0;

    virtual std::unique_ptr<Polygon> MakeCube(const glm::vec3& location, const glm::vec3& scale) const = 0;
    virtual std::unique_ptr<Polygon> MakeCube(const glm::vec3& location,
                                              const glm::vec3& scale,
                                              const std::string& texture) const = 0;

    virtual std::unique_ptr<Border> MakeBorder(const glm::vec3& location, const glm::vec3& scale) const = 0;

    virtual std::unique_ptr<Grid> MakeUniverseGrid(Mesh* universe) const = 0;
    virtual std::unique_ptr<Grid> MakeRoomGrid(Mesh* room) const = 0;
    virtual std::unique_ptr<Grid> MakeGrid(Mesh* owner,
                                           const glm::ivec3& boxes_count,
                                           const glm::vec3& box_dimension) const = 0;
    virtual std::unique_ptr<GridBox> MakeRoomGridBox(const glm::vec3& coords, Grid* grid) const = 0;
    virtual std::unique_ptr<GridBox> MakeUniverseGridBox(const glm::vec3& coords, Grid* grid) const = 0;
    virtual std::unique_ptr<GridBox> MakeGridBox(const glm::vec3& coords, Grid* grid) const = 0;

    virtual CollisionEngine* MakeCollisionEngine() const = 0;
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_OBJECTS_MESHFACTORY_H_
