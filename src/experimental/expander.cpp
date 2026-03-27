#include "experimental/expander.h"


FuncReplacerVisitor::FuncReplacerVisitor(const tsl::ordered_map<std::string, CustomFuncDef> &defs) : definitions_(defs)
{
}

vec_basic FuncReplacerVisitor::transform_args(const vec_basic &args)
{
    vec_basic new_args;
    new_args.reserve(args.size());
    for (const auto &arg : args) {
        arg->accept(*this);
        new_args.push_back(result_);
    }
    return new_args;
}

void FuncReplacerVisitor::bvisit(const FunctionSymbol &x)
{
    const std::string &name = x.get_name();

    if (definitions_.count(name)) {
        const auto &def = definitions_.at(name);
        const auto &actual_args = x.get_args();

        // Ensure argument counts match
        if (actual_args.size() != def.args.size()) {
            throw std::runtime_error("Argument count mismatch for function: " + name);
        }

        // Update 2: Transform all actual arguments recursively first
        vec_basic transformed_actuals = transform_args(actual_args);

        // Update 3: Map formal args (a, b) to actual args (10, 20)
        map_basic_basic repl_map;
        for (size_t i = 0; i < def.args.size(); ++i) {
            repl_map[def.args[i]] = transformed_actuals[i];
        }

        // Apply substitution
        result_ = def.body->subs(repl_map);
    }
    else {
        auto new_args = transform_args(x.get_args());
        result_ = function_symbol(name, new_args);
    }
}

// Standard Operators Boilerplate
void FuncReplacerVisitor::bvisit(const Add &x)
{
    result_ = add(transform_args(x.get_args()));
}

void FuncReplacerVisitor::bvisit(const Mul &x)
{
    result_ = mul(transform_args(x.get_args()));
}

void FuncReplacerVisitor::bvisit(const Pow &x)
{
    x.get_base()->accept(*this);
    auto b = result_;
    x.get_exp()->accept(*this);
    auto e = result_;
    result_ = pow(b, e);
}

void FuncReplacerVisitor::bvisit(const Abs &x)
{
    x.get_arg()->accept(*this);
    result_ = abs(result_);
}

void FuncReplacerVisitor::bvisit(const Sin &x)
{
    x.get_arg()->accept(*this);
    result_ = sin(result_);
}

void FuncReplacerVisitor::bvisit(const Cos &x)
{
    x.get_arg()->accept(*this);
    result_ = cos(result_);
}

void FuncReplacerVisitor::bvisit(const Tan &x)
{
    x.get_arg()->accept(*this);
    result_ = tan(result_);
}

void FuncReplacerVisitor::bvisit(const Csc &x)
{
    x.get_arg()->accept(*this);
    result_ = csc(result_);
}

void FuncReplacerVisitor::bvisit(const Sec &x)
{
    x.get_arg()->accept(*this);
    result_ = sec(result_);
}

void FuncReplacerVisitor::bvisit(const Cot &x)
{
    x.get_arg()->accept(*this);
    result_ = cot(result_);
}

void FuncReplacerVisitor::bvisit(const ASin &x)
{
    x.get_arg()->accept(*this);
    result_ = asin(result_);
}

void FuncReplacerVisitor::bvisit(const ACos &x)
{
    x.get_arg()->accept(*this);
    result_ = acos(result_);
}

void FuncReplacerVisitor::bvisit(const ATan &x)
{
    x.get_arg()->accept(*this);
    result_ = atan(result_);
}

void FuncReplacerVisitor::bvisit(const ACsc &x)
{
    x.get_arg()->accept(*this);
    result_ = acsc(result_);
}

void FuncReplacerVisitor::bvisit(const ASec &x)
{
    x.get_arg()->accept(*this);
    result_ = asec(result_);
}

void FuncReplacerVisitor::bvisit(const ACot &x)
{
    x.get_arg()->accept(*this);
    result_ = acot(result_);
}

void FuncReplacerVisitor::bvisit(const Sinh &x)
{
    x.get_arg()->accept(*this);
    result_ = sinh(result_);
}

void FuncReplacerVisitor::bvisit(const Cosh &x)
{
    x.get_arg()->accept(*this);
    result_ = cosh(result_);
}

void FuncReplacerVisitor::bvisit(const Tanh &x)
{
    x.get_arg()->accept(*this);
    result_ = tanh(result_);
}

void FuncReplacerVisitor::bvisit(const Csch &x)
{
    x.get_arg()->accept(*this);
    result_ = csch(result_);
}

void FuncReplacerVisitor::bvisit(const Sech &x)
{
    x.get_arg()->accept(*this);
    result_ = sech(result_);
}

void FuncReplacerVisitor::bvisit(const Coth &x)
{
    x.get_arg()->accept(*this);
    result_ = coth(result_);
}

void FuncReplacerVisitor::bvisit(const ASinh &x)
{
    x.get_arg()->accept(*this);
    result_ = asinh(result_);
}

void FuncReplacerVisitor::bvisit(const ACosh &x)
{
    x.get_arg()->accept(*this);
    result_ = acosh(result_);
}

void FuncReplacerVisitor::bvisit(const ATanh &x)
{
    x.get_arg()->accept(*this);
    result_ = atanh(result_);
}

void FuncReplacerVisitor::bvisit(const ACsch &x)
{
    x.get_arg()->accept(*this);
    result_ = acsch(result_);
}

void FuncReplacerVisitor::bvisit(const ASech &x)
{
    x.get_arg()->accept(*this);
    result_ = asech(result_);
}

void FuncReplacerVisitor::bvisit(const ACoth &x)
{
    x.get_arg()->accept(*this);
    result_ = acoth(result_);
}

void FuncReplacerVisitor::bvisit(const Log &x)
{
    x.get_arg()->accept(*this);
    result_ = log(result_);
}

void FuncReplacerVisitor::bvisit(const Erf &x)
{
    x.get_arg()->accept(*this);
    result_ = erf(result_);
}

void FuncReplacerVisitor::bvisit(const Erfc &x)
{
    x.get_arg()->accept(*this);
    result_ = erfc(result_);
}

void FuncReplacerVisitor::bvisit(const Gamma &x)
{
    x.get_arg()->accept(*this);
    result_ = gamma(result_);
}

void FuncReplacerVisitor::bvisit(const LogGamma &x)
{
    x.get_arg()->accept(*this);
    result_ = loggamma(result_);
}

void FuncReplacerVisitor::bvisit(const Floor &x)
{
    x.get_arg()->accept(*this);
    result_ = floor(result_);
}

void FuncReplacerVisitor::bvisit(const Ceiling &x)
{
    x.get_arg()->accept(*this);
    result_ = ceiling(result_);
}

void FuncReplacerVisitor::bvisit(const Truncate &x)
{
    x.get_arg()->accept(*this);
    result_ = truncate(result_);
}

// Catch-all for leaf nodes (integers, symbols, etc.)
void FuncReplacerVisitor::bvisit(const Basic &x)
{
    std::ostringstream ss;
    ss << x;
    const auto xname = ss.str();
    if (definitions_.contains(xname)) {
        const auto &defn = definitions_.at(xname);
        if (0 == defn.args.size()) {
            result_ = defn.body;
            return;
        }
    }
    result_ = x.rcp_from_this();
}

void FunctionExpander::register_function(const std::string &name, const std::vector<std::string> &arg_names, const std::string &body_str)
{

    std::vector<RCP<const Symbol>> arg_syms;
    for (const auto &s : arg_names) {
        arg_syms.push_back(symbol(s));
    }

    definitions_[name] = { arg_syms, parse(body_str) };
}

std::string FunctionExpander::expand_string(const std::string &input_str) const
{
    auto expr = parse(input_str);
    FuncReplacerVisitor visitor(definitions_);
    expr->accept(visitor);
    return visitor.result_->__str__();
}

auto FunctionExpander::expand(const std::string &input_str) const
{
    auto expr = parse(input_str);
    FuncReplacerVisitor visitor(definitions_);
    expr->accept(visitor);
    return visitor.result_;
}