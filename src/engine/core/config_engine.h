/*
 *  ConfigEngine class header
 *  @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
 *
 *  ConfigEngine implements and maintains a configuration management for the program.
 *
 *  Implement Singleton Pattern which ensure a sole ConfigEngine object for the program
 */

#ifndef ENGINE_CORE_CONFIGENGINE_H_
#define ENGINE_CORE_CONFIGENGINE_H_

#include <libconfig.h++>

namespace engine {

namespace core {

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

    /* 
     *  Return sole Instance
     */
    static ConfigEngine* Instance()
    {
        /* Raw pointer because static var */
        static auto instance = new ConfigEngine;
        return instance;
    }

    /*  
     *  Init Config
     */
    static void InitConfig()
    {
        Instance()->ParseFile();
        Instance()->DefaultValues();
    }

    /*
     *  Test if a parameter exists
     */
    inline static bool IsExist(std::string key) { return Instance()->exists(key); }

    /*
     *  Parameter Accessor
     */
    template<typename T>
    static T getSetting(std::string key)
    {
        return Instance()->getValue<T>(key);
    }

    /*
     *  Parameter Mutator
     */
    template<typename T>
    static void setSetting(std::string key, libconfig::Setting::Type setting_type, T value)
    {
        Instance()->setValue(key, setting_type, value);
    }

    static void DisplayConfig()
    {
        Instance()->ParseConfig();
    }

    static void ManageProgramParameters(int argc, char* argv[]);

protected:

    /*
     *  Standard Constructors and Assignments
     *  Protected scope beacause singleton class
     */
    ConfigEngine(){};
    ConfigEngine(const ConfigEngine&) = default;
    ConfigEngine& operator=(const ConfigEngine&) = default;

private:

    /*
     *   Parse config file and set config values
     */
    void ParseFile();

    /*
     * Parse and display config
     */
    void ParseConfig() const;

    /*
     *   Set default config values if defined neither into config file,
     *   neither in program arguments.
     *   Ensure all setting parameters are defined.
     */
    void DefaultValues();

    /*
     *  Test if exists in libconfig instance
     */
    inline bool exists(std::string key) { return cfg.exists(key); }

    /* 
     *  Internal Accessor 
     */
    template<typename T>
    T getValue(std::string key) const
    {
        T ret;
        cfg.lookupValue(key, ret);
        return ret;
    }

    /*
     *  Internal Mutator
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

    /* libconfig Object */
    libconfig::Config cfg;
};

} // namespace core

} // namespace engine

#endif // ENGINE_CORE_CONFIGENGINE_H_
