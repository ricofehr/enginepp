/**
 *  @file mouse_hid_factory.c
 *  @brief Factory Class for mouse / keyboard hids
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */


#include "nextfloor/factory/mouse_hid_factory.h"

#include "nextfloor/actions/action_factory.h"
#include "nextfloor/hid/mouse_keyboard.h"
#include "nextfloor/hid/game_input_handler.h"

namespace nextfloor {

namespace factory {

std::unique_ptr<nextfloor::gameplay::InputHandler> MouseHidFactory::MakeInputHandler() const
{
    using nextfloor::hid::GameInputHandler;
    nextfloor::actions::ActionFactory action_factory;
    return std::make_unique<GameInputHandler>(action_factory);
}

std::unique_ptr<nextfloor::gameplay::HID> MouseHidFactory::MakeHid() const
{
    using nextfloor::hid::MouseKeyboard;
    return std::make_unique<MouseKeyboard>();
}

}  // namespace factory

}  // namespace nextfloor