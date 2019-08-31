/**
 *  @file mouse_hid_factory.c
 *  @brief Factory Class for mouse / keyboard hids
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */


#include "nextfloor/factory/mouse_hid_factory.h"

#include "nextfloor/hid/mouse_keyboard.h"
#include "nextfloor/hid/game_input_handler.h"

namespace nextfloor {

namespace factory {

std::unique_ptr<nextfloor::hid::InputHandler> MouseHidFactory::MakeInputHandler()
{
    using nextfloor::hid::GameInputHandler;
    return std::make_unique<GameInputHandler>();
}

std::unique_ptr<nextfloor::hid::HID> MouseHidFactory::MakeHid()
{
    using nextfloor::hid::MouseKeyboard;
    return std::make_unique<MouseKeyboard>();
}

}  // namespace factory

}  // namespace nextfloor