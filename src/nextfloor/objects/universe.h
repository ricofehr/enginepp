/**
 *  @file universe.h
 *  @brief Universe class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_UNIVERSE_H_
#define NEXTFLOOR_OBJECTS_UNIVERSE_H_

#include "nextfloor/objects/model_mesh.h"

#include "nextfloor/objects/mesh_factory.h"

namespace nextfloor {

namespace objects {

/**
 *  @class Universe
 *  @brief Define an Universe (with rooms inside), inherits Model abstract class
 */
class Universe : public ModelMesh {

public:
    Universe(std::unique_ptr<Grid> grid, std::unique_ptr<Border> border);
    ~Universe() final = default;
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_OBJECTS_UNIVERSE_H_
