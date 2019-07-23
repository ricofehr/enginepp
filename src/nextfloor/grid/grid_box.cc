/**
 *  @file grid_box.cc
 *  @brief GridBox class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/grid/grid_box.h"

namespace nextfloor {

namespace grid {


GridBox::GridBox(glm::vec3 coords, nextfloor::objects::EngineGrid* owner)
{
    coords_ = coords;
    owner_ = owner;
}

void GridBox::add(nextfloor::objects::EngineObject* object) noexcept
{
    if (IsInside(object)) {
        return;
    }
    occupants_.push_back(object);
}

void GridBox::remove(nextfloor::objects::EngineObject* object) noexcept
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

bool GridBox::IsInside(nextfloor::objects::EngineObject* object) const noexcept
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

bool GridBox::IsFilled() const noexcept
{
    return occupants_.size() != 0;
}

bool GridBox::IsFrontPositionFilled() const noexcept
{
    return owner_->IsFrontPositionFilled(coords_);
}

bool GridBox::IsRightPositionFilled() const noexcept
{
    return owner_->IsRightPositionFilled(coords_);
}

bool GridBox::IsLeftPositionFilled() const noexcept
{
    return owner_->IsLeftPositionFilled(coords_);
}

bool GridBox::IsBackPositionFilled() const noexcept
{
    return owner_->IsBackPositionFilled(coords_);
}

bool GridBox::IsFloorPositionFilled() const noexcept
{
    return owner_->IsFloorPositionFilled(coords_);
}

bool GridBox::IsRoofPositionFilled() const noexcept
{
    return owner_->IsRoofPositionFilled(coords_);
}

} // namespace grid

} // namespace nextfloor
