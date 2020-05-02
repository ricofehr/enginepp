/**
 *  @file sprite_action_factory.h
 *  @brief SpriteAction Factory Header file
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_ACTION_SPRITEACTIONFACTORY_H_
#define NEXTFLOOR_ACTION_SPRITEACTIONFACTORY_H_

#include "nextfloor/gameplay/action_factory.h"

namespace nextfloor {

namespace action {

/**
 *  @class SrpiteActionFactory
 *  @brief Concrete Action Factory for any character (player and NPJs) actions
 */
class SpriteActionFactory : public nextfloor::gameplay::ActionFactory {

public:
    std::unique_ptr<nextfloor::gameplay::Action> MakeMoveDownAction() const final;
    std::unique_ptr<nextfloor::gameplay::Action> MakeMoveLeftAction() const final;
    std::unique_ptr<nextfloor::gameplay::Action> MakeMoveRightAction() const final;
    std::unique_ptr<nextfloor::gameplay::Action> MakeMoveUpAction() const final;
    std::unique_ptr<nextfloor::gameplay::Action> MakeJumpAction() const final;
};

}  // namespace action

}  // namespace nextfloor

#endif  // NEXTFLOOR_ACTION_SPRITEACTIONFACTORY_H_
