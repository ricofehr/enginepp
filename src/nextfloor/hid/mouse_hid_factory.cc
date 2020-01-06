/**
 *  @file mouse_hid_factory.c
 *  @brief Concrete Factory Class for mouse / keyboard hids
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */


#include "nextfloor/hid/mouse_hid_factory.h"

#include <memory>

#include "nextfloor/hid/mouse_keyboard.h"
#include "nextfloor/hid/game_input_handler.h"

namespace nextfloor {

namespace hid {

MouseHidFactory::MouseHidFactory(nextfloor::gameplay::ActionFactory* action_factory,
                                 nextfloor::gameplay::RendererFactory* renderer_factory)
{
    action_factory_ = action_factory;
    renderer_factory_ = renderer_factory;
}


std::unique_ptr<nextfloor::gameplay::InputHandler> MouseHidFactory::MakeInputHandler() const
{
    auto input_handler = std::make_unique<GameInputHandler>(MakeHid());
    input_handler->InitCommands(*action_factory_);
    return input_handler;
}

std::unique_ptr<nextfloor::gameplay::HID> MouseHidFactory::MakeHid() const
{
    return std::make_unique<MouseKeyboard>(renderer_factory_->MakeSceneInput());
}

}  // namespace hid

}  // namespace nextfloor