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
    ~FileConfigParser() noexcept final;

    void Initialize() final;
    void Display() const final;
    void ManageProgramParameters(int argc, char* argv[]) final;

    /*
     * Config Accessors
     */
    float getWindowWidth() const final { return getSetting<float>("width"); }

    float getWindowHeight() const final { return getSetting<float>("height"); }

    bool isVsync() const final { return getSetting<bool>("vsync"); }

    bool isGridMode() const final { return getSetting<bool>("grid"); }

    int getExecutionDuration() const final { return getSetting<int>("execution_time"); }

    int getDebugLevel() const final { return getSetting<int>("debug"); }

    int getCollisionGranularity() const final { return getSetting<int>("granularity"); }

    int getClippingLevel() const final { return getSetting<int>("clipping"); }

    int getThreadsCount() const final
    {
        return getSetting<int>("workers_count") > 0 ? getSetting<int>("workers_count")
                                                    : tbb::task_scheduler_init::default_num_threads();
    }

    int getParallellAlgoType() const final { return getSetting<int>("parallell"); }

    bool IsCollisionDebugEnabled() const final;
    bool IsTestDebugEnabled() const final;
    bool IsAllDebugEnabled() const final;
    bool IsPerfDebugEnabled() const final;

private:
    inline bool IsExist(const std::string& key) const final { return config_.exists(key); }

    /* Parameter Accessor */
    template <typename T>
    inline T getSetting(const std::string& key) const
    {
        T value;
        config_.lookupValue(key, value);
        return value;
    }

    /* Parameter Mutator */
    template <typename T>
    inline void setSetting(const std::string& key, libconfig::Setting::Type setting_type, T&& value)
    {
        libconfig::Setting& config_root = config_.getRoot();
        if (config_root.exists(key)) {
            config_root.remove(key);
        }
        config_root.add(key, setting_type) = value;
    }

    void ParseConfigFile();
    void InitDefaultValues() final;
    void DisplayHelp(const std::string& command_name) const final;

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
