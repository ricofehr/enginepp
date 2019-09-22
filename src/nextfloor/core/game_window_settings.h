/**
 *  @file game_window_settings.h
 *  @brief Gmae Window class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_GAMEWINDOWSETTINGS_H_
#define NEXTFLOOR_CORE_GAMEWINDOWSETTINGS_H_

#include "nextfloor/core/window_settings.h"

#include "nextfloor/renderer/scene_window.h"

namespace nextfloor {

namespace core {

/**
 * @class GameWindowSettings
 * Implementation Class For Window Settings (adapter from Scene Window object with only some accessors)
 */
class GameWindowSettings : public WindowSettings {

public:
    GameWindowSettings(nextfloor::renderer::SceneWindow* window);
    ~GameWindowSettings() final = default;

    float getWidth() const final;
    float getHeight() const final;
    float getFpsFixMoveFactor() const final;
    unsigned int getMatrixId() const final;
    unsigned int getProgramId() const final;

private:
    nextfloor::renderer::SceneWindow* game_window_;
};

}  // namespace core

}  // namespace nextfloor

#endif  // NEXTFLOOR_CORE_GAMEWINDOWSETTINGS_H_
