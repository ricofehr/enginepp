/**
 *  @file game_thing_factory.h
 *  @brief GmaeThingFactory Class for universe objects
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_THING_GAMETHINGFACTORY_H_
#define NEXTFLOOR_THING_GAMETHINGFACTORY_H_

#include "nextfloor/thing/thing_factory.h"

#include "nextfloor/mesh/border_factory.h"

#include "nextfloor/mesh/polygon_factory.h"

namespace nextfloor {

namespace thing {

/**
 *  @class GameThingFactory
 *  @brief Concrete factory for Thing Objects
 */
class GameThingFactory : public ThingFactory {

public:
    GameThingFactory(nextfloor::mesh::PolygonFactory* polygon_factory, nextfloor::mesh::BorderFactory* border_factory);
    ~GameThingFactory() final = default;

    std::unique_ptr<Thing> MakeRock(const glm::vec3& location, const glm::vec3& movement) const final;
    std::unique_ptr<Thing> MakeLittleRock(const glm::vec3& location, const glm::vec3& movement) const final;

    std::vector<std::unique_ptr<Thing>> GenerateWallBricks(glm::vec3 firstpoint,
                                                           glm::vec3 lastpoint,
                                                           const glm::vec3& brick_dimension,
                                                           const std::string& texture) const final;


private:
    static constexpr float kBrickFactorPadding = 0.5f;
    static constexpr float kBrickFactorScaling = 0.5f;

    std::unique_ptr<Thing> MakeWallBrick(const glm::vec3& location,
                                         const glm::vec3& scale,
                                         const std::string& texture) const;

    nextfloor::mesh::PolygonFactory* polygon_factory_;
    nextfloor::mesh::BorderFactory* border_factory_;
};

}  // namespace thing

}  // namespace nextfloor

#endif  // NEXTFLOOR_THING_GAMETHINGFACTORY_H_
