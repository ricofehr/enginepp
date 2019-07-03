/**
 *  @file engine_grid°box.h
 *  @brief EngineGridBox interface class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_OBJECTS_ENGINEGRIDBOX_H_
#define NEXTFLOOR_OBJECTS_ENGINEGRIDBOX_H_

#include "nextfloor/objects/engine_object.h"

namespace nextfloor {

namespace objects {

/**
 *  @class EngineGridBox
 *  @brief Abstract class who defines gridbox interface object
 */
class EngineGridBox {

public:

    EngineGridBox(EngineGridBox&&) = default;

    EngineGridBox& operator=(EngineGridBox&&) = default;

    EngineGridBox(const EngineGridBox&) = delete;

    EngineGridBox& operator=(const EngineGridBox&) = delete;

    virtual ~EngineGridBox() = default;

    virtual bool IsInside(EngineObject* object) const noexcept = 0;

    virtual void add(EngineObject* object) noexcept = 0;

    virtual void remove(EngineObject* object) noexcept = 0;

    virtual void clear() noexcept = 0;

    virtual bool IsEmpty() const noexcept = 0;

protected:

    EngineGridBox() = default;
};

} // namespace objects

} // namespace nextfloor

#endif // NEXTFLOOR_OBJECTS_ENGINEGRIDBOX_H_
