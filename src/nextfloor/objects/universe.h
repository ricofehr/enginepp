/**
 *  @file universe.h
 *  @brief Universe class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_UNIVERSE_H_
#define NEXTFLOOR_OBJECTS_UNIVERSE_H_

#include "nextfloor/objects/model_mesh.h"

#include <vector>

namespace nextfloor {

namespace objects {

/**
 *  @class Universe
 *  @brief Define an Universe (with rooms inside), inherits Model abstract class
 */
class Universe : public ModelMesh {

public:
    Universe();

    Universe(Universe&&) = default;
    Universe& operator=(Universe&&) = default;
    Universe(const Universe&) = delete;
    Universe& operator=(const Universe&) = delete;

    virtual ~Universe() override = default;

    void Draw() noexcept override final;

private:
    /** filtering only rooms with display state */
    std::vector<Mesh*> display_rooms_;
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_OBJECTS_UNIVERSE_H_
