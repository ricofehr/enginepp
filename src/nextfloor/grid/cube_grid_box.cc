/**
 *  @file cube_grid_box.cc
 *  @brief Cube GridBox class file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/grid/cube_grid_box.h"

namespace nextfloor {

namespace grid {


CubeGridBox::CubeGridBox(glm::vec3 coords, nextfloor::objects::Grid* owner)
{
    coords_ = coords;
    owner_ = owner;
}

void CubeGridBox::add(nextfloor::objects::Mesh* object) noexcept
{
    if (IsInside(object)) {
        return;
    }
    occupants_.push_back(object);
}

void CubeGridBox::remove(nextfloor::objects::Mesh* object) noexcept
{
    for (auto cnt = 0; cnt < occupants_.size(); cnt++) {
        if (object == occupants_[cnt]) {
            occupants_.erase(occupants_.begin() + cnt);
            return;
        }
    }
}

void CubeGridBox::clear() noexcept
{
    occupants_.clear();
}

bool CubeGridBox::IsInside(nextfloor::objects::Mesh* object) const noexcept
{
    for (auto& occupant : occupants_) {
        if (object == occupant) {
            return true;
        }
    }
    return false;
}

bool CubeGridBox::IsEmpty() const noexcept
{
    return occupants_.size() == 0;
}

bool CubeGridBox::IsFilled() const noexcept
{
    return occupants_.size() != 0;
}

bool CubeGridBox::IsFrontPositionFilled() const noexcept
{
    return owner_->IsFrontPositionFilled(coords_);
}

bool CubeGridBox::IsRightPositionFilled() const noexcept
{
    return owner_->IsRightPositionFilled(coords_);
}

bool CubeGridBox::IsLeftPositionFilled() const noexcept
{
    return owner_->IsLeftPositionFilled(coords_);
}

bool CubeGridBox::IsBackPositionFilled() const noexcept
{
    return owner_->IsBackPositionFilled(coords_);
}

bool CubeGridBox::IsBottomPositionFilled() const noexcept
{
    return owner_->IsBottomPositionFilled(coords_);
}

bool CubeGridBox::IsTopPositionFilled() const noexcept
{
    return owner_->IsTopPositionFilled(coords_);
}

} // namespace grid

} // namespace nextfloor
