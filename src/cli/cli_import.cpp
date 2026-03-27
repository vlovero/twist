#include "cli/cli_import.h"
#include "libloader.h"
#include "python/npy_array.h"
#include "serialize.h"

extern double *add_velocities(const int nside, const double *x, const double *y, const ptrdiff_t Nx, const ptrdiff_t node, const diffusion_t &diffusion);

int seek_next_line(FILE *file)
{
    int current_char, count;

    count = 0;

    while (true) {
        current_char = fgetc(file);
        if ((current_char == EOF) || (current_char == '\n')) {
            break;
        }
        count++;
    }

    return count;
}

std::tuple<ptrdiff_t, double *, double *> load_profiles_from_npy(const std::string &path, const int nstates)
{
    npy_array_t *npy_arr;
    ptrdiff_t nx, i, j;
    double *raw_data, *t, *y;

    npy_arr = npy_array_load(path.c_str());
    if (npy_arr == NULL) {
        throw std::runtime_error(fmt::format("Could not load {} as npy file. Check if file format is correct.", path));
    }

    if (2 != npy_arr->ndim) {
        throw std::runtime_error(fmt::format("data must have two dimensions ({} provided)", npy_arr->ndim));
    }
    if (npy_arr->shape[1] != (size_t)(1 + nstates)) {
        throw std::runtime_error(fmt::format("number of columns must be {} not {}", 1 + nstates, npy_arr->shape[1]));
    }
    if ((npy_arr->elem_size != 8) || (npy_arr->typechar != 'f')) {
        throw std::runtime_error("data type must be double/float64");
    }

    raw_data = (double *)npy_arr->data;
    nx = npy_arr->shape[0];
    t = (double *)malloc(nx * sizeof(double));
    y = (double *)malloc(nx * nstates * sizeof(double));
    if (npy_arr->fortran_order) {
        memcpy(t, raw_data, nx * sizeof(double));
        for (i = 0; i < nx; i++) {
            for (j = 0; j < nstates; j++) {
                y[i * nstates + j] = raw_data[(j + 1) * nx + i];
            }
        }
    }
    else {
        for (i = 0; i < nx; i++) {
            t[i] = raw_data[i * (nstates + 1)];
            for (j = 0; j < nstates; j++) {
                y[i * nstates + j] = raw_data[i * (nstates + 1) + (j + 1)];
            }
        }
    }

    npy_array_free(npy_arr);

    return { nx, t, y };
}

std::tuple<ptrdiff_t, double *, double *> load_profiles_from_hdf5(const std::string &path, const int nstates)
{
    if (!H5::H5File::isHdf5(path)) {
        throw std::runtime_error(fmt::format("{} is not a valid hdf5 file", path));
    }

    size_t nx, ntotal;
    double *x = NULL, *y = NULL;
    H5::H5File file(path, H5F_ACC_RDONLY);
    H5::Group group = file.openGroup("data");

    deserialize(group, "x", x, nx);
    deserialize(group, "y", y, ntotal);
    if ((ntotal / nx) != (size_t)nstates) {
        throw std::runtime_error(fmt::format("number of columns in 'data/y' must be {} not {}", nstates, ntotal / nx));
    }

    return { (ptrdiff_t)nx, x, y };
}

template <const char delimiter>
std::tuple<ptrdiff_t, double *, double *> load_profiles_from_delimited(const std::string &path, const int nstates)
{
    FILE *file;
    double *x, *y;
    ptrdiff_t nx, nxcap, ntext;
    int line_length, total_read, i, tmp, line_num;
    char *text;
    std::vector<char> buff;
    std::vector<double> row;

    if (!std::filesystem::exists(path)) {
        throw std::runtime_error(fmt::format("file {} does not exist", path));
    }

    file = (FILE *)fopen(path.c_str(), "r");
    if (file == NULL) {
        throw std::runtime_error(fmt::format("could not open file {}", path));
    }
    fseek(file, 0, SEEK_END);
    ntext = ftell(file);
    text = (char *)calloc(ntext + 1, 1);
    fseek(file, 0, SEEK_SET);
    fread(text, 1, ntext, file);
    fclose(file);

    nx = 0;
    nxcap = 1000;
    x = (double *)malloc(nxcap * sizeof(double));
    y = (double *)malloc(nxcap * nstates * sizeof(double));

    total_read = 0;
    line_num = 0;
    while (true) {
        // find next newline char
        line_num++;
        line_length = 0;
        while (total_read < ntext) {
            total_read++;
            if ((text[total_read] == '\0') || (text[total_read] == '\n')) {
                break;
            }
            line_length++;
        }


        buff.reserve(total_read + 1);
        buff.clear();
        row.clear();

        // loop over row and generate doubles
        for (i = 0; i < line_length; i++) {
            tmp = text[total_read - line_length + i];
            if (tmp == delimiter) {
                // convert to double and clear buffer
                buff.emplace_back('\0');
                row.emplace_back(atof(buff.data()));
                buff.clear();
                continue;
            }
            buff.emplace_back(tmp);
        }
        // don't forget last value before new line
        if (buff.size() != 0) {
            // convert to double and clear buffer
            buff.emplace_back('\0');
            row.emplace_back(atof(buff.data()));
            buff.clear();
        }
        else {
            break;
        }

        // make sure row matches expected dims
        if (row.size() != (size_t)(nstates + 1)) {
            throw std::runtime_error(fmt::format("on line {} there were only {} entries (expected {})", line_num, row.size(), nstates + 1));
        }

        // add data to x and y;
        nx++;
        if (nxcap < nx) {
            nxcap = std::max(nx, 2 * nxcap);
            x = (double *)realloc(x, nxcap * sizeof(double));
            y = (double *)realloc(y, nxcap * nstates * sizeof(double));
        }
        x[nx - 1] = row[0];
        memcpy(&y[(nx - 1) * nstates], &row[1], nstates * sizeof(double));
    }

    fmt::println("loaded {} data points", nx);

    // make sure x goes from 0 to L
    for (i = 0; i < nx; i++) {
        x[i] -= x[0];
    }

    return { nx, x, y };
}

auto load_needed_data(const std::string &spec_path, const std::string &parameter_set, const double wave_speed)
{
    bool is_missing;
    void *lib;
    std::string lib_path;
    func_t func_ode;
    diffusion_t diffusion;
    size_t node;
    double L;
    std::vector<std::string> params;
    std::vector<double> p = { wave_speed };
    json_res_t name;
    simdjson::dom::parser parser;
    simdjson::dom::element doc;

    if (!std::filesystem::exists(spec_path)) {
        throw std::runtime_error(fmt::format("file {} does not exist", spec_path));
    }

    try {
        doc = parser.load(spec_path);
    }
    catch (simdjson::simdjson_error &e) {
        throw std::runtime_error(fmt::format("Could not parse file {} ({})", spec_path, e.what()));
    }

    // get name and libpath
    name = get_json_entry(doc, "name", false, is_missing);
    lib_path = fmt::format(".cache/models/lib/{}.so", std::string_view{ name.get_c_str() });
    if (!std::filesystem::exists(lib_path)) {
        throw std::runtime_error(fmt::format("{} model does not have a compiled library. Make sure to create it with `twist build <path to model spec>`", std::string_view{ name.get_c_str() }));
    }

    // load ode rhs (needed if estimating wave speed)
    lib = dlopen(lib_path.c_str(), RTLD_LAZY);
    func_ode = (func_t)dlsym(lib, "func_ode");

    // load misc things
    diffusion = load_diffusion(doc);
    node = entry_to_map("system", doc).size();
    L = get_json_double(doc, "spatial_period");

    // load parameter related things
    params = entry_to_list<std::string>("params", doc, false);

    {
        auto parameter_sets = get_json_object(doc, "parameter_sets");
        auto pset_obj_ref = get_json_object(parameter_sets.value(), parameter_set.c_str());

        if (pset_obj_ref.size() != params.size()) {
            throw std::runtime_error(fmt::format("size of '{}' parameter set ({}) does not match number of model paramters ({})", parameter_set, pset_obj_ref.size(), params.size()));
        }
#ifdef TEST_WAVE_NUMBER_PARAM
        p.resize(params.size() + 3);
#else
        p.resize(params.size() + 2);
#endif
        for (auto [pname, value] : pset_obj_ref) {
            auto where = std::find(params.begin(), params.end(), pname);
            if (where != params.end()) {
                p[1 + where - params.begin()] = value;
            }
            else {
                throw std::runtime_error(fmt::format("parameter '{}' does not match any paramters from ({})", pname, fmt::join(params, ", ")));
            }
        }
        // default scaling of 1
        p[params.size() + 1] = 1.0;
#ifdef TEST_WAVE_NUMBER_PARAM
        p[params.size() + 2] = 1.0;
#endif
    }

    return std::make_tuple(std::string{ name.get_c_str() }, node, diffusion, L, func_ode, p);
}

void store_initial_wave(const std::string &name, const std::string &parameter_set, const double *t, const double *u, const size_t node, const size_t Nx, const double wave_speed, const diffusion_t &diffusion)
{
    std::filesystem::create_directories(".cache/models/init_data");
    std::string file_path = fmt::format(".cache/models/init_data/{}-{}.h5", name, parameter_set);
    H5::H5File file(file_path, H5F_ACC_TRUNC);
    {
        H5::Group group(file.createGroup("meta_data"));
        const std::string program_argv{ get_program_argv() };
        const std::string cwd = std::filesystem::current_path();
        serialize<int>(group, "file_type", 1);
        serialize<char, 1>(group, "cmdline", program_argv.c_str(), { program_argv.size() });
        serialize<char, 1>(group, "directory", cwd.c_str(), { cwd.size() });
    }
    H5::Group group(file.createGroup("data"));
    serialize<double, 1>(group, "y", u, { Nx * (node + diffusion.size()) });
    serialize<size_t>(group, "nnodes", Nx);
    serialize<double>(group, "wave_speed", wave_speed);
    serialize<double, 1>(group, "t", t, { Nx });
}

double estimate_wave_speed(func_t func, const ptrdiff_t nx, const ptrdiff_t ny, const double *x, const double *y, const double *p, const diffusion_t &diffusion)
{
    double wave_speed, num, den, L, dx;
    double *x_interp, *y_interp, *G;
    double first[ny];
    ptrdiff_t i, j, nx_interp, jl, jc, jr;

    fmt::println(COLOR_YELLOW "[Warning]: estimating the wave speed from only data can lead to bad initial approximations" COLOR_RESET);

    L = x[nx - 1] - x[0];
    nx_interp = nx;
    x_interp = (double *)malloc(nx_interp * sizeof(double));
    y_interp = (double *)malloc(nx_interp * ny * sizeof(double));
    G = (double *)malloc(nx_interp * ny * sizeof(double));

    dx = L / (nx_interp - 1);
    for (i = 0; i < nx_interp; i++) {
        x_interp[i] = i * dx;
    }
    interp(y_interp, nx_interp, x_interp, x, y, nx, ny);
    for (i = 0; i < nx_interp; i++) {
        func(&y_interp[i * ny], p, &G[i * ny]);
    }

    num = 0;
    den = 0;
    for (i = 0; i < nx_interp; i++) {
        jl = ((i - 1) + nx_interp) % nx_interp;
        jc = i;
        jr = (i + 1) % nx_interp;

        // get first derivative
        for (j = 0; j < ny; j++) {
            first[j] = (y[jr * ny + j] - y[jl * ny + j]) / (2 * dx);
        }

        // get second derivative
        for (auto &[index, coeff] : diffusion) {
            G[jc * ny + index] += coeff * (y[jl * ny + index] - 2 * y[jc * ny + index] + y[jr * ny + index]) / (dx * dx);
        }
        num += inner(&G[jc * ny], first, ny);
        den += inner(first, first, ny);
    }
    wave_speed = num / den;

    free(x_interp);
    free(y_interp);

    return wave_speed;
}

int ImportArgs::run()
{
    // TODO: wave speed, flag for already including derivatives
    double *x, *y, *u;
    ptrdiff_t i, nx;
    std::tuple<long, double *, double *> res;

    // get needed info from spec file (name, diffusion, num states)
    auto [name, node, diffusion, L, func_ode, p] = load_needed_data(spec_path, parameter_set, wave_speed);
    if (derivatives_included) {
        node += diffusion.size();
    }

    if (!std::filesystem::exists(data_path)) {
        throw std::runtime_error(fmt::format("{} does not exist", data_path));
    }

    switch (file_type) {
    case TWIST::TextFileTypes::csv:
        res = load_profiles_from_delimited<','>(data_path, node);
        break;
    case TWIST::TextFileTypes::tsv:
        res = load_profiles_from_delimited<'\t'>(data_path, node);
        break;
    case TWIST::TextFileTypes::dat:
        res = load_profiles_from_delimited<' '>(data_path, node);
        break;
    case TWIST::TextFileTypes::npy:
        res = load_profiles_from_npy(data_path, node);
        break;
    case TWIST::TextFileTypes::h5:
        res = load_profiles_from_hdf5(data_path, node);
        break;
    default:
        return 1;
        break;
    }

    nx = std::get<0>(res);
    x = std::get<1>(res);
    y = std::get<2>(res);

    // make sure spatial periods match
    if (x[nx - 1] != L) {
        throw std::runtime_error("Domain length in data file does not match spatial period in JSON file");
    }

    // wave speed estimate here
    if (!std::isfinite(wave_speed)) {
        wave_speed = estimate_wave_speed(func_ode, nx, node, x, y, p.data(), diffusion);
        fmt::println("estimated wave speed: {:.8e}", wave_speed);
        if (derivatives_included) {
            throw std::runtime_error("wave speed estimation not implemented when derivatives are included");
        }
    }

    // generate profile to store
    if (derivatives_included) {
        u = y;
    }
    else {
        u = add_velocities(2, x, y, nx, node, diffusion);
    }
    for (i = 0; i < nx; i++) {
        x[i] /= L;
    }

    if (derivatives_included) {
        node -= diffusion.size();
    }
    store_initial_wave(name, parameter_set, x, u, node, nx, wave_speed, diffusion);

    free(x);
    free(y);
    if (!derivatives_included) {
        free(u);
    }

    return 0;
}

void ImportArgs::welcome()
{
    fmt::println("Import data and generate an initial guess to a traveling wave");
}