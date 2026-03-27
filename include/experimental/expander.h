#include <string>
#include <vector>

#include "tsl/ordered_map.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "symengine/add.h"
#include "symengine/basic.h"
#include "symengine/expression.h"
#include "symengine/functions.h"
#include "symengine/mul.h"
#include "symengine/parser.h"
#include "symengine/pow.h"
#include "symengine/symbol.h"
#include "symengine/visitor.h"

#pragma GCC diagnostic pop

using namespace SymEngine;

// Update 1: Structure now holds a vector of argument symbols
struct CustomFuncDef
{
    std::vector<RCP<const Symbol>> args; // e.g., {x, y}
    RCP<const Basic> body;               // e.g., x + y
};

class FuncReplacerVisitor : public BaseVisitor<FuncReplacerVisitor>
{
public:
    RCP<const Basic> result_;
    const tsl::ordered_map<std::string, CustomFuncDef> &definitions_;

    FuncReplacerVisitor(const tsl::ordered_map<std::string, CustomFuncDef> &defs);
    vec_basic transform_args(const vec_basic &args);
    void bvisit(const FunctionSymbol &x);

    // Standard Operators Boilerplate
    void bvisit(const Add &x);
    void bvisit(const Mul &x);
    void bvisit(const Pow &x);
    void bvisit(const Abs &x);
    void bvisit(const Sin &x);
    void bvisit(const Cos &x);
    void bvisit(const Tan &x);
    void bvisit(const Csc &x);
    void bvisit(const Sec &x);
    void bvisit(const Cot &x);
    void bvisit(const ASin &x);
    void bvisit(const ACos &x);
    void bvisit(const ATan &x);
    void bvisit(const ACsc &x);
    void bvisit(const ASec &x);
    void bvisit(const ACot &x);
    void bvisit(const Sinh &x);
    void bvisit(const Cosh &x);
    void bvisit(const Tanh &x);
    void bvisit(const Csch &x);
    void bvisit(const Sech &x);
    void bvisit(const Coth &x);
    void bvisit(const ASinh &x);
    void bvisit(const ACosh &x);
    void bvisit(const ATanh &x);
    void bvisit(const ACsch &x);
    void bvisit(const ASech &x);
    void bvisit(const ACoth &x);
    void bvisit(const Log &x);
    void bvisit(const Erf &x);
    void bvisit(const Erfc &x);
    void bvisit(const Gamma &x);
    void bvisit(const LogGamma &x);
    void bvisit(const Floor &x);
    void bvisit(const Ceiling &x);
    void bvisit(const Truncate &x);
    // Catch-all for leaf nodes (integers, symbols, etc.)
    void bvisit(const Basic &x);
};

class FunctionExpander
{
private:
    tsl::ordered_map<std::string, CustomFuncDef> definitions_;

public:
    // Update 4: Registration now takes a list of arg strings
    void register_function(const std::string &name, const std::vector<std::string> &arg_names, const std::string &body_str);
    std::string expand_string(const std::string &input_str) const;
    auto expand(const std::string &input_str) const;
};