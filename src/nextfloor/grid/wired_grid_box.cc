/**
 *  @file wired_grid_box.cc
 *  @brief Wired GridBox class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/grid/wired_grid_box.h"

#include <iostream>

namespace nextfloor {

namespace grid {


WiredGridBox::WiredGridBox(glm::vec3 coords, nextfloor::objects::Grid* owner)
{
    coords_ = coords;
    owner_ = owner;
}

void WiredGridBox::add(nextfloor::objects::Mesh* object) noexcept
{
    if (IsInto(object)) {
        return;
    }
    occupants_.push_back(object);
}

void WiredGridBox::remove(nextfloor::objects::Mesh* object) noexcept
{
    for (auto cnt = 0; cnt < occupants_.size(); cnt++) {
        if (occupants_[cnt] == object) {
            occupants_.erase(occupants_.begin() + cnt);
            return;
        }
    }
}

void WiredGridBox::clear() noexcept
{
    occupants_.clear();
}

std::vector<nextfloor::objects::Mesh*> WiredGridBox::other_occupants(nextfloor::objects::Mesh* object) noexcept
{
    std::vector<nextfloor::objects::Mesh*> others(0);
    for (auto& occupant : occupants_) {
        if (occupant->id() != object->id()) {
            others.push_back(occupant);
        }
    }

    return others;
}

bool WiredGridBox::IsInto(nextfloor::objects::Mesh* object) const noexcept
{
    for (auto& occupant : occupants_) {
        if (object == occupant) {
            return true;
        }
    }
    return false;
}

bool WiredGridBox::IsEmpty() const noexcept
{
    return occupants_.size() == 0;
}

bool WiredGridBox::IsFilled() const noexcept
{
    return occupants_.size() != 0;
}

bool WiredGridBox::IsFrontPositionFilled() const noexcept
{
    return owner_->IsFrontPositionFilled(coords_);
}

bool WiredGridBox::IsRightPositionFilled() const noexcept
{
    return owner_->IsRightPositionFilled(coords_);
}

bool WiredGridBox::IsBackPositionFilled() const noexcept
{
    return owner_->IsBackPositionFilled(coords_);
}

bool WiredGridBox::IsLeftPositionFilled() const noexcept
{
    return owner_->IsLeftPositionFilled(coords_);
}

bool WiredGridBox::IsBottomPositionFilled() const noexcept
{
    return owner_->IsBottomPositionFilled(coords_);
}

bool WiredGridBox::IsTopPositionFilled() const noexcept
{
    return owner_->IsTopPositionFilled(coords_);
}


}  // namespace grid

}  // namespace nextfloor
