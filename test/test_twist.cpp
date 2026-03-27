#include "argparse/argparse.hpp"
#include "cli/colors.h"
#include "fmt/core.h"
#include "shared.h"

#include <filesystem>
#include <string_view>

namespace Tests
{
    constexpr std::string_view fhn_json_source = "{\"name\": \"__twist_test_model_fhn\", \"system\": {\"v\": \"f(v) - w\", \"w\": \"eps * (v - gamma * w)\"}, \"params\": [\"alpha\", \"gamma\", \"eps\"], \"aux\": {\"f(v)\": \"v * (v - alpha) * (1 - v)\"}, \"diffusion\": {\"v\": 1.0}, \"spatial_period\": 200.0, \"rest_state\": [0, 0], \"parameter_sets\": {\"default\": {\"alpha\": 0.25, \"gamma\": 5.0, \"eps\": 0.003}}}";

    class Context
    {
        const char *program_path = nullptr;
        FILE *temp_json_file = nullptr;
        std::filesystem::path temp_file_path;
        std::vector<std::string_view> files_to_delete;

    public:
        Context(const char *program_path) : program_path(program_path)
        {
            // delete library before running test if test has already been run
            if (std::filesystem::exists(".cache/models/lib/__twist_test_model_fhn.so")) {
                std::filesystem::remove(".cache/models/lib/__twist_test_model_fhn.so");
            }
            temp_file_path = std::filesystem::temp_directory_path() / "__twist_test_model_fhn.json";

            // write source data to temp file
            temp_json_file = fopen(temp_file_path.c_str(), "w");
            fwrite(fhn_json_source.data(), 1, fhn_json_source.size(), temp_json_file);
            fclose(temp_json_file);
            temp_json_file = nullptr;
        }

        ~Context()
        {
            std::filesystem::remove(temp_file_path);
            std::filesystem::remove(".cache/models/lib/__twist_test_model_fhn.so");
            std::filesystem::remove(".cache/models/src/__twist_test_model_fhn.cpp");
            for (const auto &path : files_to_delete) {
                std::filesystem::remove(path);
            }
        }

        void addFile(const std::string_view &path)
        {
            files_to_delete.emplace_back(path);
        }

        auto program() const
        {
            return program_path;
        }

        auto cpath() const
        {
            return temp_file_path.c_str();
        }
    };

    namespace Build
    {

        int run(const Context &context)
        {
            int code = 0;


            int chr;
            FILE *process = nullptr;
            std::string cmd = fmt::format("{} build {}", context.program(), context.cpath());
            process = popen(cmd.c_str(), "r");

            if (process == NULL) {
                return -1;
            }

            while ((chr = fgetc(process)) != EOF) {
            }

            return code;
        }
    } // namespace Build

    namespace Initialize
    {
        int run(const Context &context)
        {
            int chr;
            FILE *process = nullptr;
            std::string cmd = fmt::format("{} initialize {} --time-ode 500 --time-pde 4000", context.program(), context.cpath());
            process = popen(cmd.c_str(), "r");

            if (process == NULL) {
                return -1;
            }

            while ((chr = fgetc(process)) != EOF) {
            }

            return pclose(process);
        }
    } // namespace Initialize

    namespace Solve
    {
        int run(const Context &context)
        {
            FILE *process = nullptr;
            int code, chr;
            std::ostringstream stream;
            std::string content;
            std::string cmd = fmt::format("{} solve {} --solve-init", context.program(), context.cpath());
            process = popen(cmd.c_str(), "r");

            if (process == NULL) {
                return -1;
            }

            // read output
            while ((chr = fgetc(process)) != EOF) {
                stream << chr;
            }

            code = pclose(process);

            if (code) {
                return code;
            }

            content = stream.str();

            if (code) {
                return 1;
            }
            // check for failure messages
            return content.find("failed") != std::string::npos;
        }
    } // namespace Solve

    namespace Continuation
    {
        int run(Context &context)
        {
            int chr;
            FILE *process = nullptr;
            std::string cmd = fmt::format("{} continuation {} --solve-min-nodes 20 -b -p sps --parmin 0 --parmax 1 --ds 1e-2 --dsmax 0.1 --dsmin 1e-6", context.program(), context.cpath());
            process = popen(cmd.c_str(), "r");

            if (process == NULL) {
                return -1;
            }

            context.addFile(".cache/continuation_data/__twist_test_model_fhn-default-sps-latest.h5");

            while ((chr = fgetc(process)) != EOF) {
            }

            return pclose(process);
        }
    } // namespace Continuation

    namespace Postprocess
    {
        namespace Spectrum
        {
            int run(const Context &context)
            {
                int chr;
                FILE *process = nullptr;
                std::string cmd = fmt::format("{} postprocess .cache/continuation_data/__twist_test_model_fhn-default-sps-latest.h5 -s", context.program());
                process = popen(cmd.c_str(), "r");

                if (process == NULL) {
                    return -1;
                }

                while ((chr = fgetc(process)) != EOF) {
                }

                return pclose(process);
            }
        } // namespace Spectrum

        namespace Bifurcations
        {
            int run(const Context &context)
            {
                int chr;
                FILE *process = nullptr;
                std::string cmd = fmt::format("{} postprocess .cache/continuation_data/__twist_test_model_fhn-default-sps-latest.h5 -b -w saddle hopf", context.program());
                process = popen(cmd.c_str(), "r");

                if (process == NULL) {
                    return -1;
                }

                while ((chr = fgetc(process)) != EOF) {
                }

                return pclose(process);
            }
        } // namespace Bifurcations
    } // namespace Postprocess

    void run()
    {
        int code;
        const char *program_path = get_program_raw_argv()[0];
        Context context(program_path);

        fmt::print("Testing: build ");
        fflush(stdout);
        code = Build::run(context);
        fmt::println("{}[{}]" COLOR_RESET, code ? COLOR_RED : COLOR_GREEN, code ? "FAILED" : "PASSED");
        if (code) {
            return;
        }

        fmt::print("Testing: initialize ");
        fflush(stdout);
        code = Initialize::run(context);
        fmt::println("{}[{}]" COLOR_RESET, code ? COLOR_RED : COLOR_GREEN, code ? "FAILED" : "PASSED");
        if (code) {
            return;
        }

        fmt::print("Testing: solve ");
        fflush(stdout);
        code = Solve::run(context);
        fmt::println("{}[{}]" COLOR_RESET, code ? COLOR_RED : COLOR_GREEN, code ? "FAILED" : "PASSED");

        fmt::print("Testing: continuation ");
        fflush(stdout);
        code = Continuation::run(context);
        fmt::println("{}[{}]" COLOR_RESET, code ? COLOR_RED : COLOR_GREEN, code ? "FAILED" : "PASSED");

        fmt::print("Testing: spectra ");
        fflush(stdout);
        code = Postprocess::Spectrum::run(context);
        fmt::println("{}[{}]" COLOR_RESET, code ? COLOR_RED : COLOR_GREEN, code ? "FAILED" : "PASSED");

        fmt::print("Testing: bifurcations ");
        fflush(stdout);
        code = Postprocess::Bifurcations::run(context);
        fmt::println("{}[{}]" COLOR_RESET, code ? COLOR_RED : COLOR_GREEN, code ? "FAILED" : "PASSED");
    }
} // namespace Tests

struct TestTWIST : public argparse::Args
{
    std::string &twist = arg("twist", "path to the twist executable");

    int run() override
    {
        char *args[] = { twist.data(), NULL };
        set_program_args(1, args);
        Tests::run();
        return 0;
    }
};

int main(int argc, char **argv)
{
    auto args = argparse::parse<TestTWIST>(argc, argv, true);
    return args.run();
}
