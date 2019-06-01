/**
 *  @file config.h
 *  @brief Config Header File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_CONFIG_H_
#define NEXTFLOOR_CORE_CONFIG_H_

#include <iostream>

namespace nextfloor {

namespace core {

/**
 *  @class Config
 *  @brief Abstract class for config management
 */
class Config {

public:

    Config(Config&&) = default;

    Config& operator=(Config&&) = default;

    /* Copy constructor Deleted: Ensure a sole Instance */
    Config(const Config&) = delete;

    /* Copy assignment Deleted: Ensure a sole Instance */
    Config& operator=(const Config&) = delete;

    virtual ~Config() = default;

    virtual void Initialize() = 0;

    virtual void Display() const = 0;

    virtual void ManageProgramParameters(int argc, char* argv[]) = 0;

    /*
     * Config Accessors
     */
    virtual float getWindowWidth() const = 0;
    virtual float getWindowHeight() const = 0;
    virtual bool isVsync() const = 0;
    virtual bool isGridMode() const = 0;
    virtual int getExecutionDuration() const = 0;
    virtual int getDebugLevel() const = 0;
    virtual int getCollisionGranularity() const = 0;
    virtual int getClippingLevel() const = 0;
    virtual float getObjectsLoadFrequency() const = 0;
    virtual int getRoomsCount() const = 0;
    virtual int getObjectsCount() const = 0;
    virtual int getThreadsCount() const = 0;
    virtual int getParallellAlgoType() const = 0;
    virtual int getUniverseFactoryType() const = 0;

protected:

    Config() = default;

private:

    virtual bool IsExist(std::string key) = 0;

    virtual void InitDefaultValues() = 0;

    virtual void DisplayHelp(const std::string& command_name) const = 0;
};

} // namespace core

} // namespace nextfloor

#endif // NEXTFLOOR_CORE_CONFIG_H_
