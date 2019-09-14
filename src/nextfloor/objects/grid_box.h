/**
 *  @file engine_grid°box.h
 *  @brief EngineGridBox interface class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_GRIDBOX_H_
#define NEXTFLOOR_OBJECTS_GRIDBOX_H_

#include "nextfloor/objects/mesh.h"

namespace nextfloor {

namespace objects {

/**
 *  @class GridBox
 *  @brief Abstract class who defines gridbox interface object
 */
class GridBox {

public:
    virtual ~GridBox() = default;

    virtual void add(Mesh* object) noexcept = 0;
    virtual void remove(Mesh* object) noexcept = 0;
    virtual void clear() noexcept = 0;

    virtual bool IsInto(Mesh* object) const noexcept = 0;
    virtual bool IsEmpty() const noexcept = 0;
    virtual bool IsFilled() const noexcept = 0;

    virtual int size() const noexcept = 0;
    virtual glm::vec3 coords() const noexcept = 0;
    virtual std::vector<nextfloor::objects::Mesh*> other_occupants(nextfloor::objects::Mesh* object) noexcept = 0;
    virtual std::vector<nextfloor::objects::Mesh*> occupants() noexcept = 0;

    virtual bool IsFrontPositionFilled() const noexcept = 0;
    virtual bool IsRightPositionFilled() const noexcept = 0;
    virtual bool IsLeftPositionFilled() const noexcept = 0;
    virtual bool IsBackPositionFilled() const noexcept = 0;
    virtual bool IsBottomPositionFilled() const noexcept = 0;
    virtual bool IsTopPositionFilled() const noexcept = 0;

protected:
    GridBox() = default;

    GridBox(GridBox&&) = default;
    GridBox& operator=(GridBox&&) = default;
    GridBox(const GridBox&) = delete;
    GridBox& operator=(const GridBox&) = delete;
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_OBJECTS_ENGINEGRIDBOX_H_
