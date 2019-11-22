/**
 *  @file object_factory.h
 *  @brief Abstract Factory Class for game objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PLAYGROUND_GROUNDFACTORY_H_
#define NEXTFLOOR_PLAYGROUND_GROUNDFACTORY_H_

#include <memory>
#include <string>
#include <glm/glm.hpp>

#include "nextfloor/playground/ground.h"
#include "nextfloor/mesh/dynamic_mesh.h"

namespace nextfloor {

namespace playground {

/**
 *  @class GroundFactory
 *  @brief Abstract Factory Pattern for Universe Objects
 */
class GroundFactory {

public:
    virtual ~GroundFactory() = default;

    virtual std::unique_ptr<Ground> MakeUniverse(std::vector<std::unique_ptr<Ground>> rooms) const = 0;
    virtual std::unique_ptr<Ground> MakeRoom(const glm::vec3& location,
                                             std::vector<std::unique_ptr<nextfloor::mesh::DynamicMesh>> objects) const = 0;
};

}  // namespace playground

}  // namespace nextfloor

#endif  // NEXTFLOOR_PLAYGROUND_GROUNDFACTORY_H_
