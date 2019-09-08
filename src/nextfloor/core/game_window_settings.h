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

    GameWindowSettings(GameWindowSettings&&) = default;
    GameWindowSettings& operator=(GameWindowSettings&&) = default;
    GameWindowSettings(const GameWindowSettings&) = default;
    GameWindowSettings& operator=(const GameWindowSettings&) = default;

    virtual ~GameWindowSettings() override = default;

    virtual float getWidth() override;
    virtual float getHeight() override;
    virtual float getFpsFixMoveFactor() override;
    virtual unsigned int getMatrixId() override;
    virtual unsigned int getProgramId() override;

private:
    nextfloor::renderer::SceneWindow* game_window_;
};

}  // namespace core

}  // namespace nextfloor

#endif  // NEXTFLOOR_CORE_GAMEWINDOWSETTINGS_H_
