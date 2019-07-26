/**
 *  @file config_file.h
 *  @brief FileConfig Header File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_FILECONFIG_H_
#define NEXTFLOOR_CORE_FILECONFIG_H_

#include "nextfloor/core/config.h"

#include <libconfig.h++>
#include <memory>
#include <tbb/tbb.h>

namespace nextfloor {

namespace core {

/**
 *  @class FileConfig
 *  @brief FileConfig load program configuration from a file
 */
class FileConfig : public Config {

public:

    FileConfig();

    FileConfig(FileConfig&&) = default;
    FileConfig& operator=(FileConfig&&) = default;
    FileConfig(const FileConfig&) = delete;
    FileConfig& operator=(const FileConfig&) = delete;

    virtual ~FileConfig() override final;

    virtual void Initialize() override final;
    virtual void Display() const override final;
    virtual void ManageProgramParameters(int argc, char* argv[]) override final;

    /*
     * Config Accessors
     */
    virtual float getWindowWidth() const override final
    {
        return getSetting<float>("width");
    }

    virtual float getWindowHeight() const override final
    {
        return getSetting<float>("height");
    }

    virtual bool isVsync() const override final
    {
        return getSetting<bool>("vsync");
    }

    virtual bool isGridMode() const override final
    {
        return getSetting<bool>("grid");
    }

    virtual int getExecutionDuration() const override final
    {
        return getSetting<int>("execution_time");
    }

    virtual int getDebugLevel() const override final
    {
        return getSetting<int>("debug");
    }

    virtual int getCollisionGranularity() const override final
    {
        return getSetting<int>("granularity");
    }

    virtual int getClippingLevel() const override final
    {
        return getSetting<int>("clipping");
    }

    virtual float getObjectsLoadFrequency() const override final
    {
        return getSetting<float>("load_objects_freq");
    }

    virtual int getRoomsCount() const override final
    {
        return getSetting<int>("rooms_count");
    }

    virtual int getObjectsCount() const override final
    {
        return getSetting<int>("objects_count");
    }

    virtual int getThreadsCount() const override final
    {
        return getSetting<int>("workers_count");
    }

    virtual int getParallellAlgoType() const override final
    {
        return getSetting<int>("parallell");
    }

    virtual int getUniverseFactoryType() const override final
    {
        return getSetting<int>("factory_type");
    }


private:

    inline virtual bool IsExist(std::string key) override final
    {
        return config.exists(key);
    }

    /* Parameter Accessor */
    template<typename T>
    inline T getSetting(std::string key) const
    {
        T value;
        config.lookupValue(key, value);
        return value;
    }

    /* Parameter Mutator */
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
    virtual void InitDefaultValues() override final;
    virtual void DisplayHelp(const std::string& command_name) const override final;

    void SetDefaultParallellValueIfEmpty();
    void SetDefaultParallellThreadCountValueIfEmpty();
    void SetDefaultWidthValueIfEmpty();
    void SetDefaultHeightValueIfEmpty();
    void SetDefaultObjectCountValueIfEmpty();
    void SetDefaultLoadObjectsFreqValueIfEmpty();
    void SetDefaultRoomCountValueIfEmpty();
    void SetDefaultCollisionGranularityValueIfEmpty();
    void SetDefaultClippingValueIfEmpty();
    void SetDefaultVsyncValueIfEmpty();
    void SetDefaultGridModeValueIfEmpty();
    void SetDefaultDebugVerbosityValueIfEmpty();
    void SetDefaultExecutionTimeValueIfEmpty();
    void SetDefaultUniverseFactoryValueIfEmpty();

    int GetWorkersCount() const;

    bool IsHelpParameter(const std::string& parameter_name) const;
    bool IsDisplayConfigParameter(const std::string& parameter_name) const;

    void ManageClippingParameter(const std::string& parameter_name, const std::string& parameter_value);
    void ManageDebugParameter(const std::string& parameter_name, const std::string& parameter_value);
    void ManageExecutionTimeParameter(const std::string& parameter_name, const std::string& parameter_value);
    void ManageGranularityParameter(const std::string& parameter_name, const std::string& parameter_value);
    void ManageObjectCountParameter(const std::string& parameter_name, const std::string& parameter_value);
    void ManagePrallellAlgoTypeParameter(const std::string& parameter_name, const std::string& parameter_value);
    void ManageRoomCountParameter(const std::string& parameter_name, const std::string& parameter_value);
    void ManageLoadObjectFrequencyParameter(const std::string& parameter_name, const std::string& parameter_value);
    void ManageVsyncParameter(const std::string& parameter_name, const std::string& parameter_value);
    void ManageWorkerCountParameter(const std::string& parameter_name, const std::string& parameter_value);
    void ManageUniverseFactoryTypeParameter(const std::string& parameter_name, const std::string& parameter_value);

    void EnsureCoherentWorkerSetting();

    libconfig::Config config;
};

} // namespace core

} // namespace nextfloor

#endif // NEXTFLOOR_CORE_FILECONFIG_H_
