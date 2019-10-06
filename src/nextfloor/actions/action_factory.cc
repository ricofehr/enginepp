/**
 *  @file action_factory.c
 *  @brief Factory Class for actor actions
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/actions/action_factory.h"

#include "nextfloor/actions/fire_action.h"
#include "nextfloor/actions/jump_action.h"
#include "nextfloor/actions/movedown_action.h"
#include "nextfloor/actions/moveleft_action.h"
#include "nextfloor/actions/moveright_action.h"
#include "nextfloor/actions/moveup_action.h"
#include "nextfloor/actions/run_action.h"

namespace nextfloor {

namespace actions {

std::unique_ptr<nextfloor::gameplay::Action> ActionFactory::MakeFireAction() const
{
    return std::make_unique<FireAction>();
}

std::unique_ptr<nextfloor::gameplay::Action> ActionFactory::MakeJumpAction() const
{
    return std::make_unique<JumpAction>();
}

std::unique_ptr<nextfloor::gameplay::Action> ActionFactory::MakeMoveDownAction() const
{
    return std::make_unique<MoveDownAction>();
}

std::unique_ptr<nextfloor::gameplay::Action> ActionFactory::MakeMoveLeftAction() const
{
    return std::make_unique<MoveLeftAction>();
}

std::unique_ptr<nextfloor::gameplay::Action> ActionFactory::MakeMoveRightAction() const
{
    return std::make_unique<MoveRightAction>();
}

std::unique_ptr<nextfloor::gameplay::Action> ActionFactory::MakeMoveUpAction() const
{
    return std::make_unique<MoveUpAction>();
}

std::unique_ptr<nextfloor::gameplay::Action> ActionFactory::MakeRunAction() const
{
    return std::make_unique<RunAction>();
}

}  // namespace actions

}  // namespace nextfloor