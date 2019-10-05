/**
 *  @file level.h
 *  @brief Level class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GAMEPLAY_LEVEL_H_
#define NEXTFLOOR_GAMEPLAY_LEVEL_H_

#include <memory>

#include "nextfloor/gameplay/hid.h"
#include "nextfloor/gameplay/action.h"

namespace nextfloor {

namespace gameplay {

/**
 *  @class Level
 *  @brief Abstract class who defines level interface
 */
class Level {

public:
    virtual ~Level() = default;

    virtual void toready() = 0;
    virtual void UpdateCameraOrientation(HIDPointer angles, float input_fov) = 0;
    virtual void Move() = 0;
    virtual void Draw() = 0;
    virtual void ExecutePlayerAction(Action* command, double elapsed_time) = 0;
};

}  // namespace gameplay

}  // namespace nextfloor

#endif  // NEXTFLOOR_GAMEPLAY_LEVEL_H_
