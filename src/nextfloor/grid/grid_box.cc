/**
 *  @file grid_box.cc
 *  @brief GridBox class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/grid/grid_box.h"

namespace nextfloor {

namespace grid {


GridBox::GridBox(glm::vec3 coords, EngineGrid* owner)
{
    coords_ = coords;
    owner_ = owner;
}

void GridBox::add(EngineObject* object) noexcept
{
    if (IsInside(object)) {
        return;
    }
    occupants_.push_back(object);
}

void GridBox::remove(EngineObject* object) noexcept
{
    for (auto cnt = 0; cnt < occupants_.size(); cnt++) {
        if (object == occupants_[cnt]) {
            occupants_.erase(occupants_.begin() + cnt);
            return;
        }
    }
}

void GridBox::clear() noexcept
{
    occupants_.clear();
}

bool GridBox::IsInside(EngineObject* object) const noexcept
{
    for (auto& occupant : occupants_) {
        if (object == occupant) {
            return true;
        }
    }
    return false;
}

bool GridBox::IsEmpty() const noexcept
{
    return occupants_.size() == 0;
}

} // namespace grid

} // namespace nextfloor
