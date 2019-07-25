/**
 *  @file universe_cube_grid_box.h
 *  @brief UniverseCubeGridBox class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GRID_UNIVERSECUBEGRIDBOX_H_
#define NEXTFLOOR_GRID_UNIVERSECUBEGRIDBOX_H_

#include "nextfloor/grid/cube_grid_box.h"

namespace nextfloor {

namespace grid {

/**
 *  @class UniverseCubeGridBox
 *  @brief Abstract class who defines universe grid object
 */
class UniverseCubeGridBox : public CubeGridBox {

public:

    UniverseCubeGridBox(glm::vec3 coords, nextfloor::objects::Grid* owner);
    UniverseCubeGridBox(UniverseCubeGridBox&&) = default;
    UniverseCubeGridBox& operator=(UniverseCubeGridBox&&) = default;
    UniverseCubeGridBox(const UniverseCubeGridBox&) = default;
    UniverseCubeGridBox& operator=(const UniverseCubeGridBox&) = default;
    virtual ~UniverseCubeGridBox() = default;
};

} // namespace grid

} // namespace nextfloor

#endif // NEXTFLOOR_GRID_UNIVERSECUBEGRIDBOX_H_
