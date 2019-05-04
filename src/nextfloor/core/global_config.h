/**
 *  @file global_config.h
 *  @brief GlobalConfig Header File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_GLOBALCONFIG_H_
#define NEXTFLOOR_CORE_GLOBALCONFIG_H_

#include <libconfig.h++>
#include <memory>
#include <tbb/tbb.h>

/**
 *  @namespace nextfloor
 *  @brief Common parent namespace
 */
namespace nextfloor {

/**
 *  @namespace nextfloor::core
 *  @brief Core system
 */
namespace core {

/**
 *  @class GlobalConfig
 *  @brief GlobalConfig maintains a configuration management for the program.
 */
class GlobalConfig {

public:

    /*
     *  Program Debug Constants
     */
    static constexpr int kDEBUG_QUIET = 0;
    static constexpr int kDEBUG_TEST = 1;
    static constexpr int kDEBUG_PERF = 2;
    static constexpr int kDEBUG_COLLISION = 3;
    static constexpr int kDEBUG_ALL = 4;

    GlobalConfig();

    GlobalConfig(GlobalConfig&&) = default;

    GlobalConfig& operator=(GlobalConfig&&) = default;

    /**
     *  Copy constructor Deleted
     *  Ensure a sole Instance
     */
    GlobalConfig(const GlobalConfig&) = delete;

    /**
     *  Copy assignment Deleted
     *  Ensure a sole Instance
     */
    GlobalConfig& operator=(const GlobalConfig&) = delete;

    ~GlobalConfig();

    void Initialize()
    {
        ParseConfigFile();
        InitDefaultValues();
    }

    void Display() const;

    /**
     *  Record parameters taken from the command line
     *  @param argc the number of parameters in command line
     *  @param argv the array of parameters in the command line
     */
    void ManageProgramParameters(int argc, char* argv[]);

    /**
     *  Parameter Accessor
     *  @param key the name of the parameter
     *  @return the value of the parameter in the requested type
     */
    template<typename T>
    inline T getSetting(std::string key) const
    {
        T value;
        config.lookupValue(key, value);
        return value;
    }


private:

    inline bool IsExist(std::string key) { return config.exists(key); }

    /**
     *  Parameter Mutator
     *  @param key the name of the parameter
     *  @param setting_type the type of the parameter
     *  @param value the value of the parameter
     */
    template<typename T>
    inline void setSetting(std::string key, libconfig::Setting::Type setting_type, T value)
    {
        libconfig::Setting &config_root = config.getRoot();
        if (config_root.exists(key)) {
            config_root.remove(key);
        }
        config_root.add(key, setting_type) = value;

    }

    void ParseConfigFile();

    void InitDefaultValues();

    libconfig::Config config;
};

} // namespace core

} // namespace nextfloor

#endif // NEXTFLOOR_CORE_GLOBALCONFIG_H_
