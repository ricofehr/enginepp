/**
 *  @file mouse_hid_factory.c
 *  @brief Concrete Factory Class for mouse / keyboard hids
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */


#include "nextfloor/hid/mouse_hid_factory.h"

#include "nextfloor/hid/mouse_keyboard.h"
#include "nextfloor/hid/game_input_handler.h"

namespace nextfloor {

namespace hid {

std::unique_ptr<nextfloor::gameplay::InputHandler>
  MouseHidFactory::MakeInputHandler(const nextfloor::actions::ActionFactory& action_factory,
                                    nextfloor::renderer::RendererFactory* renderer_factory) const
{
    return std::make_unique<GameInputHandler>(*this, action_factory, renderer_factory);
}

std::unique_ptr<nextfloor::gameplay::HID> MouseHidFactory::MakeHid(nextfloor::renderer::RendererFactory* renderer_factory) const
{
    return std::make_unique<MouseKeyboard>(renderer_factory);
}

}  // namespace hid

}  // namespace nextfloor