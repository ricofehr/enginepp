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

    virtual std::unique_ptr<nextfloor::hid::HID> MakeHid() = 0;
    virtual std::unique_ptr<nextfloor::hid::InputHandler> MakeInputHandler() = 0;

protected:
    HidFactory() = default;

    HidFactory(HidFactory&&) = default;
    HidFactory& operator=(HidFactory&&) = default;
    HidFactory(const HidFactory&) = default;
    HidFactory& operator=(const HidFactory&) = default;
};

}  // namespace factory

}  // namespace nextfloor

#endif  // NEXTFLOOR_FACTORY_HIDFACTORY_H_
