/**
 *  @file object_factory.h
 *  @brief Abstract Factory Class for Characters
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_CHARACTERFACTORY_H_
#define NEXTFLOOR_OBJECTS_CHARACTERFACTORY_H_

#include <memory>
#include <glm/glm.hpp>

#include "nextfloor/objects/mesh.h"
#include "nextfloor/objects/camera.h"

namespace nextfloor {

namespace objects {

/**
 *  @class CharacterFactory
 *  @brief Abstract Factory Pattern for Characters
 */
class CharacterFactory {

public:
    virtual ~CharacterFactory() = default;

    virtual std::unique_ptr<Mesh> MakePlayer(const glm::vec3& location) const = 0;
    virtual std::unique_ptr<Camera> MakeCamera(Mesh* owner) const = 0;
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_OBJECTS_CHARACTERFACTORY_H_
