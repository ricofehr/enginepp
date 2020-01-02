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
    static constexpr float kBigScale = 1.0f;
    static constexpr float kMediumScale = 0.75f;
    static constexpr float kSmallScale = 0.5f;
    static constexpr float kSmallerScale = 0.25f;

    std::string class_name() final { return "Thing"; }
};

}  // namespace thing

}  // namespace nextfloor

#endif  // NEXTFLOOR_MESH_ROCK_H_
