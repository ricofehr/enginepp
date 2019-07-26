/**
 *  @file wired_grid_box.h
 *  @brief WiredGridBox class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GRID_WIREDGRIDBOX_H_
#define NEXTFLOOR_GRID_WIREDGRIDBOX_H_

#include "nextfloor/objects/grid_box.h"

#include <glm/glm.hpp>

#include "nextfloor/objects/grid.h"
#include "nextfloor/objects/mesh.h"

namespace nextfloor {

namespace grid {

/**
 *  @class WiredGridBox
 *  @brief Abstract class who defines generic grid object
 */
class WiredGridBox : public nextfloor::objects::GridBox {

public:

    WiredGridBox(glm::vec3 coords, nextfloor::objects::Grid* owner);
    WiredGridBox(WiredGridBox&&) = default;
    WiredGridBox& operator=(WiredGridBox&&) = default;
    WiredGridBox(const WiredGridBox&) = default;
    WiredGridBox& operator=(const WiredGridBox&) = default;
    virtual ~WiredGridBox() override = default;

    virtual void add(nextfloor::objects::Mesh* object) noexcept override;
    virtual void remove(nextfloor::objects::Mesh* object) noexcept override;
    virtual void clear() noexcept override;

    virtual bool IsInside(nextfloor::objects::Mesh* object) const noexcept override;
    virtual bool IsEmpty() const noexcept override;
    virtual bool IsFilled() const noexcept override;

    virtual bool IsFrontPositionFilled() const noexcept override;
    virtual bool IsRightPositionFilled() const noexcept override;
    virtual bool IsLeftPositionFilled() const noexcept override;
    virtual bool IsBackPositionFilled() const noexcept override;
    virtual bool IsBottomPositionFilled() const noexcept override;
    virtual bool IsTopPositionFilled() const noexcept override;

    virtual int size() const noexcept override
    {
        return occupants_.size();
    }

protected:

    nextfloor::objects::Mesh* getFirstOccupant()
    {
        if (occupants_.size() == 0)
        {
            return nullptr;
        }

        return occupants_[0];
    }

private:

    std::vector<nextfloor::objects::Mesh*> occupants_;
    nextfloor::objects::Grid* owner_;
    glm::vec3 coords_;
};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_GRIDBOX_H_
