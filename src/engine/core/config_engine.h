/*
 * ConfigEngine class header
 * @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
 *
 * Implement Singleton Design Pattern which ensure a sole ConfigEngine object for the program
 */

#ifndef ENGINE_CORE_ConfigEngine_H_
#define ENGINE_CORE_ConfigEngine_H_

#include <libconfig.h++>

namespace engine {
namespace core {

class ConfigEngine {

public:

    static constexpr int kDEBUG_TEST = 1;
    static constexpr int kDEBUG_PERF = 2;
    static constexpr int kDEBUG_COLLISION = 3;
    static constexpr int kDEBUG_ALL = 4;

    static ConfigEngine *Instance()
    {
        /* Raw pointer because static var */
        static auto *instance = new ConfigEngine;
        return instance;
    }

    static void InitConfig()
    {
        Instance()->ParseFile();
        Instance()->DefaultValues();
    }

    inline static bool IsExist(std::string key) { return Instance()->exists(key); }

    template<typename T>
    static T getSetting(std::string key)
    {
        return Instance()->getValue<T>(key);
    }

    template<typename T>
    static void setSetting(std::string key, libconfig::Setting::Type setting_type, T value)
    {
        Instance()->setValue(key, setting_type, value);
    }

    static void DisplayConfig()
    {
        Instance()->ParseConfig();
    }

    static void ManageProgramParameters(int argc, char *argv[]);

protected:

    ConfigEngine(){};
    ConfigEngine(const ConfigEngine&) = default;
    ConfigEngine& operator=(const ConfigEngine&) = default;

private:

    void ParseFile();
    void ParseConfig() const;
    void DefaultValues();

    inline bool exists(std::string key) { return cfg.exists(key); }

    template<typename T>
    T getValue(std::string key) const
    {
        T ret;
        cfg.lookupValue(key, ret);
        return ret;
    }

    template<typename T>
    void setValue(std::string key, libconfig::Setting::Type setting_type, T value)
    {
        libconfig::Setting &root = cfg.getRoot();
        if (root.exists(key)) {
            root.remove(key);
        }
        root.add(key, setting_type) = value;
    }

    libconfig::Config cfg;
};

}//namespace core
}//namespace engine

#endif //ENGINE_CORE_ConfigEngine_H_
