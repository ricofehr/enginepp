/**
 *  @file config_file.cc
 *  @brief ConfigFile Class File
 *  @author Eric Fehr (ricofehr@nextdeploy.io, github: ricofehr)
 */

#include "nextfloor/core/file_config_parser.h"

#include <iostream>
#include <iomanip>
#include <sys/stat.h>
#include <tbb/tbb.h>
#include <cassert>
#include <sstream>

#include "nextfloor/core/common_services.h"
#include "nextfloor/physic/nearer_collision_engine.h"

namespace nextfloor {

namespace core {

namespace {

static bool sInstanciated = false;

std::string GetConfigFilePath()
{
    struct stat buffer;
    return stat("config/nextfloor.ini", &buffer) == 0 ? "config/nextfloor.ini" : "config/nextfloor.ini.default";
}

}  // anonymous namespace

FileConfigParser::FileConfigParser()
{
    assert(!sInstanciated);
    sInstanciated = true;
}

void FileConfigParser::Initialize()
{
    ParseConfigFile();
    InitDefaultValues();
}

void FileConfigParser::ParseConfigFile()
{
    std::string config_file = GetConfigFilePath();
    config_.readFile(config_file.c_str());
}

void FileConfigParser::InitDefaultValues()
{
    SetDefaultParallellValueIfEmpty();
    SetDefaultParallellThreadCountValueIfEmpty();
    SetDefaultWidthValueIfEmpty();
    SetDefaultHeightValueIfEmpty();
    SetDefaultCollisionGranularityValueIfEmpty();
    SetDefaultClippingValueIfEmpty();
    SetDefaultVsyncValueIfEmpty();
    SetDefaultGridModeValueIfEmpty();
    SetDefaultDebugVerbosityValueIfEmpty();
    SetDefaultExecutionTimeValueIfEmpty();
}

void FileConfigParser::SetDefaultParallellValueIfEmpty()
{
    using nextfloor::physic::NearerCollisionEngine;

    if (!IsExist("parallell")) {
        setSetting("parallell", libconfig::Setting::TypeInt, NearerCollisionEngine::kPARALLELL_SERIAL);
    }
}

void FileConfigParser::SetDefaultParallellThreadCountValueIfEmpty()
{
    if (!IsExist("workers_count")) {
        setSetting("workers_count", libconfig::Setting::TypeInt, tbb::task_scheduler_init::default_num_threads());
    }
}

void FileConfigParser::SetDefaultWidthValueIfEmpty()
{
    if (!IsExist("width")) {
        setSetting("width", libconfig::Setting::TypeFloat, 800.0f);
    }
}

void FileConfigParser::SetDefaultHeightValueIfEmpty()
{
    if (!IsExist("height")) {
        setSetting("height", libconfig::Setting::TypeFloat, 600.0f);
    }
}

void FileConfigParser::SetDefaultCollisionGranularityValueIfEmpty()
{
    if (!IsExist("granularity")) {
        setSetting("granularity", libconfig::Setting::TypeInt, 16);
    }
}

void FileConfigParser::SetDefaultClippingValueIfEmpty()
{
    if (!IsExist("clipping")) {
        setSetting("clipping", libconfig::Setting::TypeInt, 0);
    }
}

void FileConfigParser::SetDefaultVsyncValueIfEmpty()
{
    if (!IsExist("vsync")) {
        setSetting("vsync", libconfig::Setting::TypeBoolean, true);
    }
}

void FileConfigParser::SetDefaultGridModeValueIfEmpty()
{
    if (!IsExist("grid")) {
        setSetting("grid", libconfig::Setting::TypeBoolean, false);
    }
}

void FileConfigParser::SetDefaultDebugVerbosityValueIfEmpty()
{
    if (!IsExist("debug")) {
        setSetting("debug", libconfig::Setting::TypeInt, 0);
    }
}

void FileConfigParser::SetDefaultExecutionTimeValueIfEmpty()
{
    if (!IsExist("execution_time")) {
        setSetting("execution_time", libconfig::Setting::TypeInt, 0);
    }
}

void FileConfigParser::Display() const
{
    auto count_workers = getThreadsCount();

    std::cout << "Parallell mode (1 -> serial, 2 -> tbb, 3 -> opencl): " << getSetting<int>("parallell") << std::endl;
    std::cout << "Window width: " << getSetting<float>("width") << std::endl;
    std::cout << "Window height: " << getSetting<float>("height") << std::endl;
    std::cout << "NearerCollisionEngine granularity: " << getSetting<int>("granularity") << std::endl;
    std::cout << "Clipping (0 -> no clipping, 1 -> high clipping, 2 -> low clipping): " << getSetting<int>("clipping")
              << std::endl;
    std::cout << "Workers count: " << count_workers << std::endl;
    std::cout << "Execution Time (0 -> no limit): " << getSetting<int>("execution_time") << std::endl;
    std::cout << "Vsync (limit framerate to monitor): " << getSetting<bool>("vsync") << std::endl;
    std::cout << "WiredGrid mode (not fill polygons): " << getSetting<bool>("grid") << std::endl;
    std::cout << "Debug mode (0 -> no debug, 1 -> test debug, 2 -> performance debug, 3 -> "
                 "collision debug, 4 -> all debug): "
              << getSetting<int>("debug") << std::endl;
}

void FileConfigParser::ManageProgramParameters(int argc, char* argv[])
{
    const std::string program_name(argv[0]);
    auto is_display_config = false;
    auto cnt = 1;

    while (cnt < argc) {

        const std::string parameter_name(argv[cnt++]);

        if (IsHelpParameter(parameter_name)) {
            DisplayHelp(program_name);
            CommonServices::getExit()->ExitOnSuccess();
        }

        if (IsDisplayConfigParameter(parameter_name)) {
            is_display_config = true;
            continue;
        }

        /* name/value parameters need one more word */
        assert(cnt < argc);
        const std::string parameter_value(argv[cnt++]);

        ManageClippingParameter(parameter_name, parameter_value);
        ManageDebugParameter(parameter_name, parameter_value);
        ManageExecutionTimeParameter(parameter_name, parameter_value);
        ManageGranularityParameter(parameter_name, parameter_value);
        ManagePrallellAlgoTypeParameter(parameter_name, parameter_value);
        ManageVsyncParameter(parameter_name, parameter_value);
        ManageWorkerCountParameter(parameter_name, parameter_value);
    }

    EnsureCoherentWorkerSetting();

    if (is_display_config) {
        Display();
        CommonServices::getExit()->ExitOnSuccess();
    }
}

bool FileConfigParser::IsDisplayConfigParameter(const std::string& parameter_name) const
{
    if (parameter_name == "-l") {
        return true;
    }
    return false;
}

bool FileConfigParser::IsHelpParameter(const std::string& parameter_name) const
{
    if (parameter_name == "-h") {
        return true;
    }
    return false;
}

void FileConfigParser::DisplayHelp(const std::string& command_name) const
{
    std::cout << command_name << " can be used with following options who overrides config file" << std::endl;
    std::cout << "-c n   Clipping, 0: no clipping, 1: high clipping, 2: low clipping" << std::endl;
    std::cout << "-d n   Debug mode, 0: no debug, 1: test debug, 2: performance debug, 3: "
                 "collision debug, 4: all debug"
              << std::endl;
    std::cout << "-e n   Execution Time, 0: no limit" << std::endl;
    std::cout << "-g n   Granularity on collision computes" << std::endl;
    std::cout << "-h     Display help" << std::endl;
    std::cout << "-l 1|0 Enable/Disable display config" << std::endl;
    std::cout << "-p serial|tbb|opencl" << std::endl
              << "       serial: no parallellism" << std::endl
              << "       tbb: uses intel tbb library" << std::endl
              << "       opencl: uses opencl for collision computes" << std::endl;
    std::cout << "-v 1|0 Enable/Disable vsync" << std::endl;
    std::cout << "-w n   Workers (cpu core) count (disabled if -p serial), "
              << "0: no limit, all cpu cores" << std::endl;
}

void FileConfigParser::ManageClippingParameter(const std::string& parameter_name, const std::string& parameter_value)
{
    if (parameter_name == "-c") {
        setSetting("clipping", libconfig::Setting::TypeInt, std::stoi(parameter_value));
    }
}

void FileConfigParser::ManageDebugParameter(const std::string& parameter_name, const std::string& parameter_value)
{
    if (parameter_name == "-d") {
        setSetting("debug", libconfig::Setting::TypeInt, std::stoi(parameter_value));
    }
}

void FileConfigParser::ManageExecutionTimeParameter(const std::string& parameter_name, const std::string& parameter_value)
{
    if (parameter_name == "-e") {
        setSetting("execution_time", libconfig::Setting::TypeInt, std::stoi(parameter_value));
    }
}

void FileConfigParser::ManageGranularityParameter(const std::string& parameter_name, const std::string& parameter_value)
{
    if (parameter_name == "-g") {
        setSetting("granularity", libconfig::Setting::TypeInt, std::stoi(parameter_value));
    }
}

void FileConfigParser::ManagePrallellAlgoTypeParameter(const std::string& parameter_name,
                                                       const std::string& parameter_value)
{
    using nextfloor::physic::NearerCollisionEngine;

    if (parameter_name == "-p") {
        if (parameter_value == "serial") {
            setSetting("parallell", libconfig::Setting::TypeInt, NearerCollisionEngine::kPARALLELL_SERIAL);
        }

        if (parameter_value == "tbb") {
            setSetting("parallell", libconfig::Setting::TypeInt, NearerCollisionEngine::kPARALLELL_TBB);
        }

        if (parameter_value == "opencl") {
            setSetting("parallell", libconfig::Setting::TypeInt, NearerCollisionEngine::kPARALLELL_CL);
        }
    }
}

void FileConfigParser::ManageVsyncParameter(const std::string& parameter_name, const std::string& parameter_value)
{
    if (parameter_name == "-v") {
        setSetting("vsync", libconfig::Setting::TypeBoolean, std::stoi(parameter_value) == 1);
    }
}

void FileConfigParser::ManageWorkerCountParameter(const std::string& parameter_name, const std::string& parameter_value)
{
    if (parameter_name == "-w") {
        setSetting("workers_count", libconfig::Setting::TypeInt, std::stoi(parameter_value));
    }
}

void FileConfigParser::EnsureCoherentWorkerSetting()
{
    using nextfloor::physic::NearerCollisionEngine;

    /* Disable tbb usage when serial parallel algorithm is setted */
    if (getSetting<int>("parallell") == NearerCollisionEngine::kPARALLELL_SERIAL) {
        setSetting("workers_count", libconfig::Setting::TypeInt, 1);
    }

    /* Manage Threads Parallelism */
    if (getThreadsCount()) {
        tbb_threads_config_ = std::make_unique<tbb::task_scheduler_init>(getThreadsCount());
    }
}

bool FileConfigParser::IsCollisionDebugEnabled() const
{
    return getDebugLevel() >= CommonServices::getLog()->kDEBUG_COLLISION;
}

bool FileConfigParser::IsTestDebugEnabled() const
{
    return getDebugLevel() >= CommonServices::getLog()->kDEBUG_TEST;
}

bool FileConfigParser::IsAllDebugEnabled() const
{
    return getDebugLevel() >= CommonServices::getLog()->kDEBUG_ALL;
}

bool FileConfigParser::IsPerfDebugEnabled() const
{
    return getDebugLevel() >= CommonServices::getLog()->kDEBUG_PERF;
}

FileConfigParser::~FileConfigParser() noexcept
{
    assert(sInstanciated);
    sInstanciated = false;
}

}  // namespace core

}  // namespace nextfloor
