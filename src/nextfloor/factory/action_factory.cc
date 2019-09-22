/**
 *  @file action_factory.c
 *  @brief Factory Class for actor actions
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */
#include "nextfloor/factory/action_factory.h"

#include "nextfloor/actions/fire_action.h"
#include "nextfloor/actions/jump_action.h"
#include "nextfloor/actions/movedown_action.h"
#include "nextfloor/actions/moveleft_action.h"
#include "nextfloor/actions/moveright_action.h"
#include "nextfloor/actions/moveup_action.h"
#include "nextfloor/actions/run_action.h"

namespace nextfloor {

namespace factory {

std::unique_ptr<nextfloor::actions::Action> ActionFactory::MakeFireAction() const
{
    using nextfloor::actions::FireAction;
    return std::make_unique<FireAction>();
}

std::unique_ptr<nextfloor::actions::Action> ActionFactory::MakeJumpAction() const
{
    using nextfloor::actions::JumpAction;
    return std::make_unique<JumpAction>();
}

std::unique_ptr<nextfloor::actions::Action> ActionFactory::MakeMoveDownAction() const
{
    using nextfloor::actions::MoveDownAction;
    return std::make_unique<MoveDownAction>();
}

std::unique_ptr<nextfloor::actions::Action> ActionFactory::MakeMoveLeftAction() const
{
    using nextfloor::actions::MoveLeftAction;
    return std::make_unique<MoveLeftAction>();
}

std::unique_ptr<nextfloor::actions::Action> ActionFactory::MakeMoveRightAction() const
{
    using nextfloor::actions::MoveRightAction;
    return std::make_unique<MoveRightAction>();
}

std::unique_ptr<nextfloor::actions::Action> ActionFactory::MakeMoveUpAction() const
{
    using nextfloor::actions::MoveUpAction;
    return std::make_unique<MoveUpAction>();
}

std::unique_ptr<nextfloor::actions::Action> ActionFactory::MakeRunAction() const
{
    using nextfloor::actions::RunAction;
    return std::make_unique<RunAction>();
}

}  // namespace factory

}  // namespace nextfloor