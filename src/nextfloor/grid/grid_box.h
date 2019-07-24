/**
 *  @file grid_box.h
 *  @brief GridBox class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GRID_GRIDBOX_H_
#define NEXTFLOOR_GRID_GRIDBOX_H_

#include "nextfloor/objects/engine_grid_box.h"

#include <glm/glm.hpp>

#include "nextfloor/objects/engine_grid.h"
#include "nextfloor/objects/engine_object.h"

namespace nextfloor {

namespace grid {

/**
 *  @class GridBox
 *  @brief Abstract class who defines generic grid object
 */
class GridBox : public nextfloor::objects::EngineGridBox {

public:

    GridBox(glm::vec3 coords, nextfloor::objects::EngineGrid* owner);
    GridBox(GridBox&&) = default;
    GridBox& operator=(GridBox&&) = default;
    GridBox(const GridBox&) = default;
    GridBox& operator=(const GridBox&) = default;
    virtual ~GridBox() override = default;

    virtual void add(nextfloor::objects::EngineObject* object) noexcept override;
    virtual void remove(nextfloor::objects::EngineObject* object) noexcept override;
    virtual void clear() noexcept override;

    virtual bool IsInside(nextfloor::objects::EngineObject* object) const noexcept override;
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

    nextfloor::objects::EngineObject* getFirstOccupant()
    {
        if (occupants_.size() == 0)
        {
            return nullptr;
        }

        return occupants_[0];
    }

private:

    std::vector<nextfloor::objects::EngineObject*> occupants_;
    nextfloor::objects::EngineGrid* owner_;
    glm::vec3 coords_;
};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_GRIDBOX_H_
