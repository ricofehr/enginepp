/**
 *  @file action.h
 *  @brief BaseClass Action class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_ACTIONS_ACTION_H_
#define NEXTFLOOR_ACTIONS_ACTION_H_

#include "nextfloor/objects/mesh.h"

namespace nextfloor {

namespace actions {

/**
 *  @class Action
 *  @brief Action pure interface, baseclass for apply Action Pattern to player and NPJ actions
 */
class Action {

public:
    virtual ~Action() = default;
    /**
     * Execute Action on target object (actor)
     */
    virtual void execute(nextfloor::objects::Mesh* actor) = 0;
};

}  // namespace actions

}  // namespace nextfloor

#endif  // NEXTFLOOR_ACTIONS_ACTION_H_
