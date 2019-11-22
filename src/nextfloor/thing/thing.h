/**
 *  @file rock.h
 *  @brief Rock class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_MESH_ROCK_H_
#define NEXTFLOOR_MESH_ROCK_H_

#include "nextfloor/mesh/dynamic_mesh.h"

namespace nextfloor {

namespace thing {

/**
 *  @class Thing
 *  @brief Thing
 */
class Thing : public nextfloor::mesh::DynamicMesh {

public:
    static constexpr float kBIG_SCALE = 1.0f;
    static constexpr float kMEDIUM_SCALE = 0.75f;
    static constexpr float kSMALL_SCALE = 0.5f;
    static constexpr float kSMALLER_SCALE = 0.5f;

    std::string class_name() final { return "Thing"; }
};

}  // namespace thing

}  // namespace nextfloor

#endif  // NEXTFLOOR_MESH_ROCK_H_
