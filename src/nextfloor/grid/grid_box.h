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

using nextfloor::objects::EngineGridBox;
using nextfloor::objects::EngineGrid;
using nextfloor::objects::EngineObject;

/**
 *  @class GridBox
 *  @brief Abstract class who defines generic grid object
 */
class GridBox : public EngineGridBox {

public:

    GridBox(glm::vec3 coords, EngineGrid* owner);

    GridBox(GridBox&&) = default;

    GridBox& operator=(GridBox&&) = default;

    GridBox(const GridBox&) = default;

    GridBox& operator=(const GridBox&) = default;

    virtual ~GridBox() override = default;

    virtual bool IsInside(EngineObject* object) const noexcept override;

    virtual void add(EngineObject* object) noexcept override;

    virtual void remove(EngineObject* object) noexcept override;

    virtual void clear() noexcept override;

    virtual bool IsEmpty() const noexcept override;

    //std::vector<EngineObject*> occupants() const noexcept { return occupants_; }
    virtual int size() const noexcept override
    {
        return occupants_.size();
    }

protected:

    EngineObject* getFirstOccupant()
    {
        if (occupants_.size() == 0)
        {
            return nullptr;
        }

        return occupants_[0];
    }

private:

    glm::vec3 coords_;
    EngineGrid* owner_;
    std::vector<EngineObject*> occupants_;
};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_GRIDBOX_H_
