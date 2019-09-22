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
 * Base Pure interface For Window Settings (adapter from Scene Window object with only some accessors)
 */
class WindowSettings {

public:
    virtual ~WindowSettings() = default;

    virtual float getWidth() const = 0;
    virtual float getHeight() const = 0;
    virtual float getFpsFixMoveFactor() const = 0;
    virtual unsigned int getMatrixId() const = 0;
    virtual unsigned int getProgramId() const = 0;
};

}  // namespace core

}  // namespace nextfloor

#endif  // NEXTFLOOR_CORE_WINDOWSETTINGS_H_
