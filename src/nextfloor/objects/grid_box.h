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

class Grid;

/**
 *  @class GridBox
 *  @brief Abstract class who defines gridbox interface object
 */
class GridBox {

public:
    virtual ~GridBox() = default;

    virtual void set_owner(Grid* owner) = 0;
    virtual void add(Mesh* object) = 0;
    virtual void remove(Mesh* object) = 0;
    virtual void clear() = 0;

    virtual bool IsInto(const Mesh& object) const = 0;
    virtual bool IsEmpty() const = 0;
    virtual bool IsFilled() const = 0;

    virtual int size() const = 0;
    virtual glm::vec3 coords() const = 0;
    virtual std::vector<nextfloor::objects::Mesh*> other_occupants(const nextfloor::objects::Mesh& object) const = 0;
    virtual std::vector<nextfloor::objects::Mesh*> occupants() const = 0;

    virtual bool IsFrontPositionFilled() const = 0;
    virtual bool IsRightPositionFilled() const = 0;
    virtual bool IsLeftPositionFilled() const = 0;
    virtual bool IsBackPositionFilled() const = 0;
    virtual bool IsBottomPositionFilled() const = 0;
    virtual bool IsTopPositionFilled() const = 0;
};

}  // namespace objects

}  // namespace nextfloor

#endif  // NEXTFLOOR_OBJECTS_ENGINEGRIDBOX_H_
