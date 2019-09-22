/**
 *  @file hid_factory.h
 *  @brief Abstract Factory Class Implementation for hids
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_FACTORY_HIDFACTORY_H_
#define NEXTFLOOR_FACTORY_HIDFACTORY_H_

#include "nextfloor/hid/hid.h"
#include "nextfloor/hid/input_handler.h"

namespace nextfloor {

namespace factory {

/**
 *  @class HidFactory
 *  @brief Abstract Factory for hids (human interface devices)
 */
class HidFactory {

public:
    virtual ~HidFactory() = default;

    virtual std::unique_ptr<nextfloor::hid::HID> MakeHid() const = 0;
    virtual std::unique_ptr<nextfloor::hid::InputHandler> MakeInputHandler() const = 0;
};

}  // namespace factory

}  // namespace nextfloor

#endif  // NEXTFLOOR_FACTORY_HIDFACTORY_H_
