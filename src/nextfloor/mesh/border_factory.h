/**
 *  @file border_factory.h
 *  @brief Abstract Factory Class for Border elements
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_MESH_BORDERFACTORY_H_
#define NEXTFLOOR_MESH_BORDERFACTORY_H_

#include <memory>
#include <glm/glm.hpp>

#include "nextfloor/mesh/border.h"

namespace nextfloor {

namespace mesh {

/**
 *  @class BorderFactory
 *  @brief Abstract Factory Pattern for Border elements
 */
class BorderFactory {

public:
    virtual ~BorderFactory() = default;

    virtual std::unique_ptr<Border> MakeBorder(const glm::vec3& location, float scale) const = 0;
    virtual std::unique_ptr<Border> MakeBorder(const glm::vec3& location, const glm::vec3& scale) const = 0;
};

}  // namespace mesh

}  // namespace nextfloor

#endif  // NEXTFLOOR_MESH_PHYSICFACTORY_H_
