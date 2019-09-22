/**
 *  @file mouse_hid_factory.h
 *  @brief Abstract Factory Class Implementation for mouse / keyboard hids
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_FACTORY_MOUSEHIDFACTORY_H_
#define NEXTFLOOR_FACTORY_MOUSEHIDFACTORY_H_

#include "nextfloor/factory/hid_factory.h"

namespace nextfloor {

namespace factory {

/**
 *  @class MouseHidFactory
 *  @brief Abstract Factory for mouse / keyboard hids (human interface devices)
 */
class MouseHidFactory : public HidFactory {

public:
    std::unique_ptr<nextfloor::hid::InputHandler> MakeInputHandler() const final;
    std::unique_ptr<nextfloor::hid::HID> MakeHid() const final;
};

}  // namespace factory

}  // namespace nextfloor

#endif  // NEXTFLOOR_FACTORY_HIDFACTORY_H_
