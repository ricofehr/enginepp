/**
 *  @file wired_grid_box.cc
 *  @brief Wired GridBox class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/grid/wired_grid_box.h"

#include <iostream>

namespace nextfloor {

namespace grid {


WiredGridBox::WiredGridBox(const glm::vec3& coords)
{
    coords_ = coords;
}

void WiredGridBox::set_owner(nextfloor::objects::Grid* owner)
{
    owner_ = owner;
}

void WiredGridBox::add(nextfloor::objects::Mesh* object)
{
    if (IsInto(*object)) {
        return;
    }
    occupants_.push_back(object);
}

void WiredGridBox::remove(nextfloor::objects::Mesh* object)
{
    for (auto cnt = 0; cnt < occupants_.size(); cnt++) {
        if (occupants_[cnt] == object) {
            occupants_.erase(occupants_.begin() + cnt);
            return;
        }
    }
}

void WiredGridBox::clear()
{
    occupants_.clear();
}

std::vector<nextfloor::objects::Mesh*> WiredGridBox::other_occupants(const nextfloor::objects::Mesh& object) const
{
    std::vector<nextfloor::objects::Mesh*> others(0);
    for (const auto& occupant : occupants_) {
        if (occupant->id() != object.id()) {
            others.push_back(occupant);
        }
    }

    return others;
}

bool WiredGridBox::IsInto(const nextfloor::objects::Mesh& object) const
{
    for (const auto& occupant : occupants_) {
        if (object.id() == occupant->id()) {
            return true;
        }
    }
    return false;
}

bool WiredGridBox::IsEmpty() const
{
    return occupants_.size() == 0;
}

bool WiredGridBox::IsFilled() const
{
    return occupants_.size() != 0;
}

bool WiredGridBox::IsFrontPositionFilled() const
{
    assert(owner_ != nullptr);
    return owner_->IsFrontPositionFilled(coords_);
}

bool WiredGridBox::IsRightPositionFilled() const
{
    assert(owner_ != nullptr);
    return owner_->IsRightPositionFilled(coords_);
}

bool WiredGridBox::IsBackPositionFilled() const
{
    assert(owner_ != nullptr);
    return owner_->IsBackPositionFilled(coords_);
}

bool WiredGridBox::IsLeftPositionFilled() const
{
    assert(owner_ != nullptr);
    return owner_->IsLeftPositionFilled(coords_);
}

bool WiredGridBox::IsBottomPositionFilled() const
{
    assert(owner_ != nullptr);
    return owner_->IsBottomPositionFilled(coords_);
}

bool WiredGridBox::IsTopPositionFilled() const
{
    assert(owner_ != nullptr);
    return owner_->IsTopPositionFilled(coords_);
}


}  // namespace grid

}  // namespace nextfloor
