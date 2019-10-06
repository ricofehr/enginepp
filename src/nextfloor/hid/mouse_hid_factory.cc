/**
 *  @file mouse_hid_factory.c
 *  @brief Concrete Factory Class for mouse / keyboard hids
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */


#include "nextfloor/hid/mouse_hid_factory.h"

#include "nextfloor/hid/mouse_keyboard.h"
#include "nextfloor/hid/game_input_handler.h"
#include "nextfloor/actions/sprite_action_factory.h"

namespace nextfloor {

namespace hid {

std::unique_ptr<nextfloor::gameplay::InputHandler> MouseHidFactory::MakeInputHandler() const
{
    nextfloor::actions::SpriteActionFactory action_factory;
    return std::make_unique<GameInputHandler>(*this, action_factory);
}

std::unique_ptr<nextfloor::gameplay::HID> MouseHidFactory::MakeHid() const
{
    return std::make_unique<MouseKeyboard>();
}

}  // namespace hid

}  // namespace nextfloor