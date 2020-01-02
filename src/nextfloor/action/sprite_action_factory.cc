/**
 *  @file action_factory.c
 *  @brief Factory Class for actor actions
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/action/sprite_action_factory.h"

#include "nextfloor/action/movedown_action.h"
#include "nextfloor/action/moveleft_action.h"
#include "nextfloor/action/moveright_action.h"
#include "nextfloor/action/moveup_action.h"

namespace nextfloor {

namespace action {

std::unique_ptr<nextfloor::gameplay::Action> SpriteActionFactory::MakeMoveDownAction() const
{
    return std::make_unique<MoveDownAction>();
}

std::unique_ptr<nextfloor::gameplay::Action> SpriteActionFactory::MakeMoveLeftAction() const
{
    return std::make_unique<MoveLeftAction>();
}

std::unique_ptr<nextfloor::gameplay::Action> SpriteActionFactory::MakeMoveRightAction() const
{
    return std::make_unique<MoveRightAction>();
}

std::unique_ptr<nextfloor::gameplay::Action> SpriteActionFactory::MakeMoveUpAction() const
{
    return std::make_unique<MoveUpAction>();
}

}  // namespace action

}  // namespace nextfloor