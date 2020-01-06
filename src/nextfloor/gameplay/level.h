/**
 *  @file level.h
 *  @brief Level class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GAMEPLAY_LEVEL_H_
#define NEXTFLOOR_GAMEPLAY_LEVEL_H_

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

    virtual void UpdateCameraOrientation(HIDPointer angles) = 0;
    virtual void Move() = 0;
    virtual void Draw(float window_size_ratio) = 0;
    virtual void ExecutePlayerAction(Action* command, double elapsed_time) = 0;
};

}  // namespace gameplay

}  // namespace nextfloor

#endif  // NEXTFLOOR_GAMEPLAY_LEVEL_H_
