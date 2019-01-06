/**
 *  @file config_engine.h
 *  @brief ConfigEngine Header File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_CONFIGENGINE_H_
#define NEXTFLOOR_CORE_CONFIGENGINE_H_

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
 *  @class ConfigEngine
 *  @brief ConfigEngine maintains a configuration management for the program.\n
 *  Implement Singleton Pattern which ensure a sole ConfigEngine object for the program
 */
class ConfigEngine {

public:

    /*
     *  Debug Constants
     */
    static constexpr int kDEBUG_QUIET = 0;
    static constexpr int kDEBUG_TEST = 1;
    static constexpr int kDEBUG_PERF = 2;
    static constexpr int kDEBUG_COLLISION = 3;
    static constexpr int kDEBUG_ALL = 4;

    /**
     *  Default Move constructor
     */
    ConfigEngine(ConfigEngine&&) = default;

    /**
     *  Default Move assignment
     */
    ConfigEngine& operator=(ConfigEngine&&) = default;

    /**
     *  Copy constructor Deleted
     *  Ensure a sole Instance
     */
    ConfigEngine(const ConfigEngine&) = delete;

    /**
     *  Copy assignment Deleted
     *  Ensure a sole Instance
     */
    ConfigEngine& operator=(const ConfigEngine&) = delete;

    /**
     *  Return sole Instance
     *  @return sole ConfigEngine instance
     */
    static ConfigEngine* Instance()
    {
        /* Raw pointer because static var */
        static auto instance = new ConfigEngine;
        return instance;
    }

    /**
     *  Init Config
     */
    static void InitConfig()
    {
        Instance()->ParseFile();
        Instance()->DefaultValues();
    }

    /**
     *  Test if a parameter exists
     *  @param key the name of the parameter
     *  @return true if the parameter exists in the config, false in the other case
     */
    inline static bool IsExist(std::string key) { return Instance()->exists(key); }

    /**
     *  Display all config parameter in the current state
     */
    static void DisplayConfig()
    {
        Instance()->ParseConfig();
    }

    /**
     *  Record parameters taken from the command line
     *  @param argc the number of parameters in command line
     *  @param argv the array of parameters in the command line
     */
    static void ManageProgramParameters(int argc, char* argv[]);

    /**
     *  Parameter Accessor
     *  @param key the name of the parameter
     *  @return the value of the parameter in the requested type
     */
    template<typename T>
    static T getSetting(std::string key)
    {
        return Instance()->getValue<T>(key);
    }

    /**
     *  Parameter Mutator
     *  @param key the name of the parameter
     *  @param setting_type the type of the parameter
     *  @param value the value of the parameter
     */
    template<typename T>
    static void setSetting(std::string key, libconfig::Setting::Type setting_type, T value)
    {
        Instance()->setValue(key, setting_type, value);
    }

protected:

    /**
     *  Default Constructor
     *  With a protected scope for avoid any directly Allocation
     */
    ConfigEngine() = default;

private:

    /**
     *   Parse config file and set config values
     */
    void ParseFile();

    /**
     * Parse and display config
     */
    void ParseConfig() const;

    /**
     *   Set default config values if defined neither into config file,
     *   neither in program arguments.
     *   Ensure all setting parameters are defined.
     */
    void DefaultValues();

    /**
     *  Test if parameter exists in libconfig instance
     *  @param key the name of the parameter
     */
    inline bool exists(std::string key) const { return cfg.exists(key); }

    /**
     *  Internal Accessor
     *  @param key the name of the parameter
     *  @return the value of the parameter in the requested type
     */
    template<typename T>
    T getValue(std::string key) const
    {
        T ret;
        cfg.lookupValue(key, ret);
        return ret;
    }

    /**
     *  Internal Mutator
     *  @param key the name of the parameter
     *  @param setting_type the type of the parameter
     *  @param value the value of the parameterin the requested type
     */
    template<typename T>
    void setValue(std::string key, libconfig::Setting::Type setting_type, T value)
    {
        libconfig::Setting &root = cfg.getRoot();
        if (root.exists(key)) {
            root.remove(key);
        }
        root.add(key, setting_type) = value;
    }


    /** libconfig Object */
    libconfig::Config cfg;
};

} // namespace core

} // namespace nextfloor

#endif // NEXTFLOOR_CORE_CONFIGENGINE_H_
