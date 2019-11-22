/**
 *  @file wired_grid_box.cc
 *  @brief Wired GridBox class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/layout/wired_grid_box.h"

#include <iostream>

namespace nextfloor {

namespace layout {


WiredGridBox::WiredGridBox(const glm::vec3& coords)
{
    coords_ = coords;
}

void WiredGridBox::set_owner(nextfloor::playground::Grid* owner)
{
    owner_ = owner;
}

void WiredGridBox::add(nextfloor::mesh::Mesh* object)
{
    assert(object != nullptr);

    if (IsInto(*object)) {
        return;
    }

    tbb::mutex::scoped_lock lock_map(mutex_);
    occupants_.push_back(object);
}

void WiredGridBox::remove(nextfloor::mesh::Mesh* object)
{
    tbb::mutex::scoped_lock lock_map(mutex_);

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

std::vector<nextfloor::mesh::Mesh*> WiredGridBox::occupants() const
{
    std::vector<nextfloor::mesh::Mesh*> occupants;
    for (auto& occupant : occupants_) {
        auto meshes = occupant->AllStubMeshs();
        occupants.insert(occupants.end(), meshes.begin(), meshes.end());
    }

    return occupants;
}

std::vector<nextfloor::mesh::Mesh*> WiredGridBox::other_occupants(const nextfloor::mesh::Mesh& object) const
{
    std::vector<nextfloor::mesh::Mesh*> others(0);
    for (const auto& occupant : occupants_) {
        if (occupant->id() != object.id()) {
            others.push_back(occupant);
        }
    }

    return others;
}

nextfloor::mesh::Mesh* WiredGridBox::getFirstOccupant()
{
    if (occupants_.size() == 0) {
        return nullptr;
    }

    return occupants_[0];
}

bool WiredGridBox::IsInto(const nextfloor::mesh::Mesh& object) const
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


}  // namespace layout

}  // namespace nextfloor
