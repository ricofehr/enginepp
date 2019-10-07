/**
 *  @file mouse_factory.h
 *  @brief Cincrete Factory Class Implementation for mouse / keyboard hids
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_HID_MOUSEHIDFACTORY_H_
#define NEXTFLOOR_HID_MOUSEHIDFACTORY_H_

#include "nextfloor/gameplay/hid_factory.h"

#include "nextfloor/gameplay/action_factory.h"
#include "nextfloor/gameplay/renderer_factory.h"

namespace nextfloor {

namespace hid {

/**
 *  @class MouseFactory
 *  @brief Concrete Factory for mouse / keyboard hids (human interface devices)
 */
class MouseHidFactory : public nextfloor::gameplay::HidFactory {

public:
    MouseHidFactory(nextfloor::gameplay::ActionFactory* action_factory,
                    nextfloor::gameplay::RendererFactory* renderer_factory);
    ~MouseHidFactory() final = default;

    std::unique_ptr<nextfloor::gameplay::InputHandler> MakeInputHandler() const final;
    std::unique_ptr<nextfloor::gameplay::HID> MakeHid() const final;

private:
    nextfloor::gameplay::ActionFactory* action_factory_;
    nextfloor::gameplay::RendererFactory* renderer_factory_;
};

}  // namespace hid

}  // namespace nextfloor

#endif  // NEXTFLOOR_HID_MOUSEHIDFACTORY_H_
