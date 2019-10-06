/**
 *  @file mouse_factory.h
 *  @brief Cincrete Factory Class Implementation for mouse / keyboard hids
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_HID_MOUSEHIDFACTORY_H_
#define NEXTFLOOR_HID_MOUSEHIDFACTORY_H_

#include "nextfloor/hid/hid_factory.h"

namespace nextfloor {

namespace hid {

/**
 *  @class MouseFactory
 *  @brief Concrete Factory for mouse / keyboard hids (human interface devices)
 */
class MouseHidFactory : public HidFactory {

public:
    std::unique_ptr<nextfloor::gameplay::InputHandler>
      MakeInputHandler(const nextfloor::actions::ActionFactory& action_factory,
                       nextfloor::renderer::RendererFactory* renderer_factory) const final;
    std::unique_ptr<nextfloor::gameplay::HID> MakeHid(nextfloor::renderer::RendererFactory* renderer_factory) const final;
};

}  // namespace hid

}  // namespace nextfloor

#endif  // NEXTFLOOR_HID_MOUSEHIDFACTORY_H_
