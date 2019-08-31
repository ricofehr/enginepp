/**
 *  @file fire_action.h
 *  @brief Fire action for an object in the world
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_ACTIONS_FIREACTION_H_
#define NEXTFLOOR_ACTIONS_FIREACTION_H_

#include "nextfloor/actions/action.h"

namespace nextfloor {

namespace actions {

/**
 *  @class FireAction
 *  @brief Implements Action (Command Pattern), generate Fire action for any object
 */
class FireAction : public Action {

public:
    FireAction() = default;

    FireAction(FireAction&&) = default;
    FireAction& operator=(FireAction&&) = default;
    FireAction(const FireAction&) = default;
    FireAction& operator=(const FireAction&) = default;

    virtual ~FireAction() override = default;

    void execute(nextfloor::objects::Mesh* actor) override final;
};

}  // namespace actions

}  // namespace nextfloor

#endif  // NEXTFLOOR_ACTIONS_FIREACTION_H_
