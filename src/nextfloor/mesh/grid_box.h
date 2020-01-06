/**
 *  @file engine_grid°box.h
 *  @brief EngineGridBox interface class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_PLAYGROUND_GRIDBOX_H_
#define NEXTFLOOR_PLAYGROUND_GRIDBOX_H_

#include <vector>
#include <glm/glm.hpp>

#include "nextfloor/mesh/mesh.h"

#include "nextfloor/playground/grid.h"

namespace nextfloor {

namespace mesh {

/**
 *  @class GridBox
 *  @brief Abstract class who defines gridbox interface object
 */
class GridBox {

public:
    virtual ~GridBox() = default;

    virtual void set_owner(nextfloor::playground::Grid* owner) = 0;
    virtual void add(Mesh* object) = 0;
    virtual void remove(Mesh* object) = 0;
    virtual void clear() = 0;

    virtual bool IsInto(const Mesh& object) const = 0;
    virtual bool IsEmpty() const = 0;
    virtual bool IsFilled() const = 0;

    virtual int size() const = 0;
    virtual glm::vec3 coords() const = 0;
    virtual std::vector<Mesh*> other_occupants(const Mesh& object) const = 0;
    virtual std::vector<Mesh*> occupants() const = 0;

    virtual bool IsFrontPositionFilled() const = 0;
    virtual bool IsRightPositionFilled() const = 0;
    virtual bool IsLeftPositionFilled() const = 0;
    virtual bool IsBackPositionFilled() const = 0;
    virtual bool IsBottomPositionFilled() const = 0;
    virtual bool IsTopPositionFilled() const = 0;
};

}  // namespace mesh

}  // namespace nextfloor

#endif  // NEXTFLOOR_PLAYGROUND_ENGINEGRIDBOX_H_
