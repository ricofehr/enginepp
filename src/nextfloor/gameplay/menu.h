/**
 *  @file menu.h
 *  @brief MENU class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_GAMEPLAY_MENU_H_
#define NEXTFLOOR_GAMEPLAY_MENU_H_

namespace nextfloor {

namespace gameplay {

/**
 *  @class RoomGrid
 *  @brief Abstract class who defines grid for Rooms
 */
class Menu {

public:
    virtual ~Menu() = default;

    virtual void Enable() = 0;
    virtual void Disable() = 0;
    virtual void MenuLoop() = 0;
    virtual void Init(void* glfw_window) = 0;
    virtual bool IsResumeGamePressed() = 0;
    virtual bool IsExitGamePressed() = 0;
};

}  // namespace hud

}  // namespace nextfloor

#endif  // NEXTFLOOR_GAMEPLAY_MENU_H_
