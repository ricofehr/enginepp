/**
 *  @file window_settings.h
 *  @brief WindowSettings class
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_WINDOWSETTINGS_H_
#define NEXTFLOOR_CORE_WINDOWSETTINGS_H_

namespace nextfloor {

namespace core {

/**
 * @class WindowSettings
 * Base Class For Window Settings (adapter from Scene Window object with only some accessors)
 */
class WindowSettings {

public:
    virtual ~WindowSettings() = default;

    virtual float getWidth() = 0;
    virtual float getHeight() = 0;
    virtual float getFpsFixMoveFactor() = 0;
    virtual unsigned int getMatrixId() = 0;
    virtual unsigned int getProgramId() = 0;

protected:
    WindowSettings() = default;

    WindowSettings(WindowSettings&&) = default;
    WindowSettings& operator=(WindowSettings&&) = default;
    WindowSettings(const WindowSettings&) = default;
    WindowSettings& operator=(const WindowSettings&) = default;
};

}  // namespace core

}  // namespace nextfloor

#endif  // NEXTFLOOR_CORE_WINDOWSETTINGS_H_
