/**
 *  @file config_file_parser.h
 *  @brief FileConfigParser Header File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#ifndef NEXTFLOOR_CORE_FILECONFIGPARSER_H_
#define NEXTFLOOR_CORE_FILECONFIGPARSER_H_

#include "nextfloor/core/config_parser.h"

#include <libconfig.h++>
#include <memory>
#include <tbb/tbb.h>

namespace nextfloor {

namespace core {

/**
 *  @class FileConfigParser
 *  @brief FileConfigParser load program configuration from a file
 */
class FileConfigParser : public ConfigParser {

public:
    FileConfigParser();

    FileConfigParser(FileConfigParser&&) = default;
    FileConfigParser& operator=(FileConfigParser&&) = default;
    FileConfigParser(const FileConfigParser&) = delete;
    FileConfigParser& operator=(const FileConfigParser&) = delete;

    virtual ~FileConfigParser() override final;

    virtual void Initialize() override final;
    virtual void Display() const override final;
    virtual void ManageProgramParameters(int argc, char* argv[]) override final;

    /*
     * Config Accessors
     */
    virtual float getWindowWidth() const override final { return getSetting<float>("width"); }

    virtual float getWindowHeight() const override final { return getSetting<float>("height"); }

    virtual bool isVsync() const override final { return getSetting<bool>("vsync"); }

    virtual bool isGridMode() const override final { return getSetting<bool>("grid"); }

    virtual int getExecutionDuration() const override final { return getSetting<int>("execution_time"); }

    virtual int getDebugLevel() const override final { return getSetting<int>("debug"); }

    virtual int getCollisionGranularity() const override final { return getSetting<int>("granularity"); }

    virtual int getClippingLevel() const override final { return getSetting<int>("clipping"); }

    virtual int getThreadsCount() const override final
    {
        return getSetting<int>("workers_count") > 0 ? getSetting<int>("workers_count")
                                                    : tbb::task_scheduler_init::default_num_threads();
    }

    virtual int getParallellAlgoType() const override final { return getSetting<int>("parallell"); }

    virtual bool IsCollisionDebugEnabled() const override final;
    virtual bool IsTestDebugEnabled() const override final;
    virtual bool IsAllDebugEnabled() const override final;
    virtual bool IsPerfDebugEnabled() const override final;

private:
    inline virtual bool IsExist(std::string key) override final { return config_.exists(key); }

    /* Parameter Accessor */
    template<typename T>
    inline T getSetting(std::string key) const
    {
        T value;
        config_.lookupValue(key, value);
        return value;
    }

    /* Parameter Mutator */
    template<typename T>
    inline void setSetting(std::string key, libconfig::Setting::Type setting_type, T value)
    {
        libconfig::Setting& config_root = config_.getRoot();
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
    void SetDefaultCollisionGranularityValueIfEmpty();
    void SetDefaultClippingValueIfEmpty();
    void SetDefaultVsyncValueIfEmpty();
    void SetDefaultGridModeValueIfEmpty();
    void SetDefaultDebugVerbosityValueIfEmpty();
    void SetDefaultExecutionTimeValueIfEmpty();

    bool IsHelpParameter(const std::string& parameter_name) const;
    bool IsDisplayConfigParameter(const std::string& parameter_name) const;

    void ManageClippingParameter(const std::string& parameter_name, const std::string& parameter_value);
    void ManageDebugParameter(const std::string& parameter_name, const std::string& parameter_value);
    void ManageExecutionTimeParameter(const std::string& parameter_name, const std::string& parameter_value);
    void ManageGranularityParameter(const std::string& parameter_name, const std::string& parameter_value);
    void ManagePrallellAlgoTypeParameter(const std::string& parameter_name, const std::string& parameter_value);
    void ManageVsyncParameter(const std::string& parameter_name, const std::string& parameter_value);
    void ManageWorkerCountParameter(const std::string& parameter_name, const std::string& parameter_value);

    void EnsureCoherentWorkerSetting();

    libconfig::Config config_;
    std::unique_ptr<tbb::task_scheduler_init> tbb_threads_config_{nullptr};
};

}  // namespace core

}  // namespace nextfloor

#endif  // NEXTFLOOR_CORE_FILECONFIGPARSER_H_
