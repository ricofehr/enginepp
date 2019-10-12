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

    virtual std::unique_ptr<Mesh> MakeRock(const glm::vec3& location, const glm::vec3& movement) const = 0;
    virtual std::unique_ptr<Mesh> MakeLittleRock(const glm::vec3& location, const glm::vec3& movement) const = 0;
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_OBJECTS_MESHFACTORY_H_
