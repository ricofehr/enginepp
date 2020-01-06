/**
 *  @file thing_factory.h
 *  @brief ThingFactory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_THING_THINGFACTORY_H_
#define NEXTFLOOR_THING_THINGFACTORY_H_

#include <memory>
#include <vector>
#include <string>
#include <glm/glm.hpp>

#include "nextfloor/thing/thing.h"

namespace nextfloor {

namespace thing {

/**
 *  @class ThingFactory
 *  @brief Concret factory for Universe Objects
 */
class ThingFactory {

public:
    virtual ~ThingFactory() = default;

    virtual std::unique_ptr<Thing> MakeRock(const glm::vec3& location, const glm::vec3& movement) const = 0;
    virtual std::unique_ptr<Thing> MakeLittleRock(const glm::vec3& location, const glm::vec3& movement) const = 0;

    virtual std::vector<std::unique_ptr<Thing>> GenerateWallBricks(glm::vec3 firstpoint,
                                                                   glm::vec3 lastpoint,
                                                                   const glm::vec3& brick_dimension,
                                                                   const std::string& texture) const = 0;
};

}  // namespace thing

}  // namespace nextfloor

#endif  // NEXTFLOOR_THING_THINGFACTORY_H_