/**
 *  @file config_parser.h
 *  @brief ConfigParser Header File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_CONFIGPARSER_H_
#define NEXTFLOOR_CORE_CONFIGPARSER_H_

#include <iostream>

namespace nextfloor {

namespace core {

/**
 *  @class ConfigParser
 *  @brief Abstract class for config management
 */
class ConfigParser {

public:

    ConfigParser(ConfigParser&&) = default;
    ConfigParser& operator=(ConfigParser&&) = default;
    ConfigParser(const ConfigParser&) = delete;
    ConfigParser& operator=(const ConfigParser&) = delete;

    virtual ~ConfigParser() = default;

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
    virtual int getThreadsCount() const = 0;
    virtual int getParallellAlgoType() const = 0;
    virtual bool IsCollisionDebugEnabled() const = 0;
    virtual bool IsTestDebugEnabled() const = 0;
    virtual bool IsAllDebugEnabled() const = 0;
    virtual bool IsPerfDebugEnabled() const = 0;

protected:

    ConfigParser() = default;

private:

    virtual bool IsExist(std::string key) = 0;
    virtual void InitDefaultValues() = 0;
    virtual void DisplayHelp(const std::string& command_name) const = 0;
};

} // namespace core

} // namespace nextfloor

#endif // NEXTFLOOR_CORE_CONFIGPARSER_H_
