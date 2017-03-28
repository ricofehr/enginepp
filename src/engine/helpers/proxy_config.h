/*
* ProxyConfig class header
* @author Eric Fehr (ricofehr@nextdeploy.io, @github: ricofehr)
*/

#ifndef ENGINE_HELPERS_PROXYCONFIG_H_
#define ENGINE_HELPERS_PROXYCONFIG_H_

#include <libconfig.h++>

namespace engine {
namespace helpers {

class ProxyConfig {

public:
    static void InitConfig()
    {
        get().ParseFile();
        get().DefaultValues();
    }

    inline static bool IsExist(std::string key) { return get().exists(key); }

    template<typename T>
    static T getSetting(std::string key)
    {
        return get().getValue<T>(key);
    }

    template<typename T>
    static void setSetting(std::string key, libconfig::Setting::Type setting_type, T value)
    {
        get().setValue(key, setting_type, value);
    }

    static void DisplayConfig()
    {
        get().ParseConfig();
    }

    static void ManageProgramParameters(int argc, char *argv[]);

    static constexpr int kDEBUG_TEST = 1;
    static constexpr int kDEBUG_PERF = 2;
    static constexpr int kDEBUG_COLLISION = 3;
    static constexpr int kDEBUG_ALL = 4;

private:
    libconfig::Config cfg;

    ProxyConfig(){};
    ProxyConfig(const ProxyConfig&);
    ProxyConfig& operator=(const ProxyConfig&);
    void ParseFile();
    void ParseConfig() const;
    void DefaultValues();

    static ProxyConfig& get()
    {
        static ProxyConfig instance;
        return instance;
    }

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
        if (root.exists(key))
            root.remove(key);
        root.add(key, setting_type) = value;
    }
};

}//namespace helpers
}//namespace engine

#endif //ENGINE_HELPERS_PROXYCONFIG_H_
