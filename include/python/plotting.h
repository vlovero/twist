#ifndef TWIST_PYTHON_PLOTTING_H
#define TWIST_PYTHON_PLOTTING_H

// #include "Python.h"
#include "shared.h"
#include <cassert>
#include <string>
#include <vector>

void solve_plot(bool plot_after_solve_init, bool plot_dense, bool spectrum, int N, int node, int ncol, int nnodes, double *t, double *tc, double *y, double *td, double *yd, const std::vector<double> &alphar, const std::vector<double> &alphai, const std::vector<double> &beta, const std::vector<std::string> &var_names);

// NEW
int run_python_script(const std::string_view &script, const std::vector<std::string> &args, const bool escape = true);

namespace python::plot
{
    void set_script(const std::string &script_path);
    const std::string_view get_script();
} // namespace python::plot

namespace python::interim
{
    void set_script(const std::string &script_path);
    const std::string_view get_script();
} // namespace python::interim

namespace python::animate
{
    void set_script(const std::string &script_path);
    const std::string_view get_script();
} // namespace python::animate

namespace python::visualize
{
    void set_script(const std::string &script_path);
    const std::string_view get_script();
} // namespace python::visualize

namespace python::simviz
{
    void set_script(const std::string &script_path);
    const std::string_view get_script();
} // namespace python::simviz

#endif // TWIST_PYTHON_PLOTTING_H