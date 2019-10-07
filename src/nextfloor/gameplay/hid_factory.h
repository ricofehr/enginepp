/**
 *  @file hid_factory.h
 *  @brief Abstract Factory Class Implementation for hids
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GAMEPLAY_HIDFACTORY_H_
#define NEXTFLOOR_GAMEPLAY_HIDFACTORY_H_

#include "nextfloor/gameplay/hid.h"
#include "nextfloor/gameplay/input_handler.h"

namespace nextfloor {

namespace gameplay {

/**
 *  @class HidFactory
 *  @brief Abstract Factory for hids (human interface devices)
 */
class HidFactory {

public:
    virtual ~HidFactory() = default;

    virtual std::unique_ptr<InputHandler> MakeInputHandler() const = 0;
    virtual std::unique_ptr<HID> MakeHid() const = 0;
};

}  // namespace gameplay

}  // namespace nextfloor

#endif  // NEXTFLOOR_GAMEPLAY_HIDFACTORY_H_
