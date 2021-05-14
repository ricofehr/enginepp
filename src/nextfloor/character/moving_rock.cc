/**
 *  @file moving_rock.cc
 *  @brief MovingRock class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/character/moving_rock.h"

#include <cassert>
#include <utility>

namespace nextfloor {

namespace character {

MovingRock::MovingRock(std::unique_ptr<nextfloor::mesh::Border> border, std::unique_ptr<FSM> fsm,
                       std::vector<std::unique_ptr<nextfloor::mesh::Polygon>> rock,
                       const glm::vec3& movement)
{
    border_ = std::move(border);
    fsm_ = std::move(fsm);
    fsm_->set_owner(this);

    polygons_ = std::move(rock);
    set_movement(movement);
}

}  // namespace character

}  // namespace nextfloor
