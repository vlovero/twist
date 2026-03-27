#include "cxxcodeprinter.h"
namespace SymEngine
{

    void CXXCodePrinter::bvisit(const Basic &x)
    {
        (void)x;
        std::ostringstream s;
        s << "Not supported: ";
        s << __PRETTY_FUNCTION__;
        throw SymEngineException(s.str());
    }

    void CXXCodePrinter::bvisit(const Complex &x)
    {
        (void)x;
        std::ostringstream s;
        s << "I";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const Interval &x)
    {
        std::string var = str_;
        std::ostringstream s;
        bool is_inf = eq(*x.get_start(), *NegInf);
        if (not is_inf) {
            s << var;
            if (x.get_left_open()) {
                s << " > ";
            }
            else {
                s << " >= ";
            }
            s << apply(x.get_start());
        }
        if (neq(*x.get_end(), *Inf)) {
            if (not is_inf) {
                s << " && ";
            }
            s << var;
            if (x.get_right_open()) {
                s << " < ";
            }
            else {
                s << " <= ";
            }
            s << apply(x.get_end());
        }
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const Contains &x)
    {
        x.get_expr()->accept(*this);
        x.get_set()->accept(*this);
    }

    void CXXCodePrinter::bvisit(const Piecewise &x)
    {
        std::ostringstream s;
        auto vec = x.get_vec();
        for (size_t i = 0;; ++i) {
            if (i == vec.size() - 1) {
                if (neq(*vec[i].second, *boolTrue)) {
                    throw SymEngineException("Code generation requires a (Expr, True) at the end");
                }
                s << "(\n   " << apply(vec[i].first) << "\n";
                break;
            }
            else {
                s << "((";
                s << apply(vec[i].second);
                s << ") ? (\n   ";
                s << apply(vec[i].first);
                s << "\n)\n: ";
            }
        }
        for (size_t i = 0; i < vec.size(); i++) {
            s << ")";
        }
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const Rational &x)
    {
        std::ostringstream o;
        double n = mp_get_d(get_num(x.as_rational_class()));
        double d = mp_get_d(get_den(x.as_rational_class()));
        o << print_double(n) << "/" << print_double(d);
        str_ = o.str();
    }

    void CXXCodePrinter::bvisit(const Reals &x)
    {
        (void)x;
        std::ostringstream s;
        s << "Not supported: ";
        s << __PRETTY_FUNCTION__;
        throw SymEngineException(s.str());
    }

    void CXXCodePrinter::bvisit(const Rationals &x)
    {
        (void)x;
        std::ostringstream s;
        s << "Not supported: ";
        s << __PRETTY_FUNCTION__;
        throw SymEngineException(s.str());
    }

    void CXXCodePrinter::bvisit(const Integers &x)
    {
        (void)x;
        std::ostringstream s;
        s << "Not supported: ";
        s << __PRETTY_FUNCTION__;
        throw SymEngineException(s.str());
    }

    void CXXCodePrinter::bvisit(const EmptySet &x)
    {
        (void)x;
        std::ostringstream s;
        s << "Not supported: ";
        s << __PRETTY_FUNCTION__;
        throw SymEngineException(s.str());
    }

    void CXXCodePrinter::bvisit(const FiniteSet &x)
    {
        (void)x;
        std::ostringstream s;
        s << "Not supported: ";
        s << __PRETTY_FUNCTION__;
        throw SymEngineException(s.str());
    }

    void CXXCodePrinter::bvisit(const UniversalSet &x)
    {
        (void)x;
        std::ostringstream s;
        s << "Not supported: ";
        s << __PRETTY_FUNCTION__;
        throw SymEngineException(s.str());
    }

    void CXXCodePrinter::bvisit(const Abs &x)
    {
        std::ostringstream s;
        s << NAMESPACE_STR "abs(" << apply(x.get_arg()) << ")";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const Ceiling &x)
    {
        std::ostringstream s;
        s << NAMESPACE_STR "ceil(" << apply(x.get_arg()) << ")";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const Floor &x)
    {
        std::ostringstream s;
        s << NAMESPACE_STR "floor(" << apply(x.get_arg()) << ")";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const Truncate &x)
    {
        std::ostringstream s;
        s << NAMESPACE_STR "trunc(" << apply(x.get_arg()) << ")";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const Sin &x)
    {
        std::ostringstream s;
        s << NAMESPACE_STR "sin(" << apply(x.get_arg()) << ")";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const Cos &x)
    {
        std::ostringstream s;
        s << NAMESPACE_STR "cos(" << apply(x.get_arg()) << ")";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const Tan &x)
    {
        std::ostringstream s;
        s << NAMESPACE_STR "tan(" << apply(x.get_arg()) << ")";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const Sinh &x)
    {
        std::ostringstream s;
        s << NAMESPACE_STR "sinh(" << apply(x.get_arg()) << ")";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const Cosh &x)
    {
        std::ostringstream s;
        s << NAMESPACE_STR "cosh(" << apply(x.get_arg()) << ")";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const Tanh &x)
    {
        std::ostringstream s;
        s << NAMESPACE_STR "tanh(" << apply(x.get_arg()) << ")";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const Csc &x)
    {
        std::ostringstream s;
        s << NAMESPACE_STR "(1.0 / (std::sin(" << apply(x.get_arg()) << "))";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const Sec &x)
    {
        std::ostringstream s;
        s << NAMESPACE_STR "(1.0 / (std::cos(" << apply(x.get_arg()) << "))";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const Cot &x)
    {
        std::ostringstream s;
        s << NAMESPACE_STR "(1.0 / (std::tan(" << apply(x.get_arg()) << "))";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const Csch &x)
    {
        std::ostringstream s;
        s << NAMESPACE_STR "(1.0 / (std::sinh(" << apply(x.get_arg()) << "))";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const Sech &x)
    {
        std::ostringstream s;
        s << NAMESPACE_STR "(1.0 / (std::cosh(" << apply(x.get_arg()) << "))";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const Coth &x)
    {
        std::ostringstream s;
        s << NAMESPACE_STR "(1.0 / (std::tanh(" << apply(x.get_arg()) << "))";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const Max &x)
    {
        std::ostringstream s;
        const auto &args = x.get_args();
        switch (args.size()) {
        case 0:
        case 1:
            throw SymEngineException("Impossible");
        case 2:
            s << NAMESPACE_STR "fmax(" << apply(args[0]) << ", " << apply(args[1]) << ")";
            break;
        default: {
            vec_basic inner_args(args.begin() + 1, args.end());
            auto inner = max(inner_args);
            s << NAMESPACE_STR "fmax(" << apply(args[0]) << ", " << apply(inner) << ")";
            break;
        }
        }
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const Min &x)
    {
        std::ostringstream s;
        const auto &args = x.get_args();
        switch (args.size()) {
        case 0:
        case 1:
            throw SymEngineException("Impossible");
        case 2:
            s << NAMESPACE_STR "fmin(" << apply(args[0]) << ", " << apply(args[1]) << ")";
            break;
        default: {
            vec_basic inner_args(args.begin() + 1, args.end());
            auto inner = min(inner_args);
            s << NAMESPACE_STR "fmin(" << apply(args[0]) << ", " << apply(inner) << ")";
            break;
        }
        }
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const Pow &x)
    {
        std::ostringstream s;
        const auto &args = x.get_args();
        switch (args.size()) {
        case 0:
        case 1:
            throw SymEngineException("Impossible");
        case 2:
            // s << NAMESPACE_STR "pow(" << apply(args[0]) << ", " << apply(args[1]) << ")";
            _print_pow(s, args[0], args[1]);
            break;
        default: {
            vec_basic inner_args(args.begin() + 1, args.end());
            auto inner = max(inner_args);
            s << NAMESPACE_STR "pow(" << apply(args[0]) << ", " << apply(inner) << ")";
            break;
        }
        }
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const Constant &x)
    {
        if (eq(x, *E)) {
            str_ = "M_E";
        }
        else if (eq(x, *pi)) {
            str_ = "M_PI";
        }
        else {
            str_ = x.get_name();
        }
    }

    void CXXCodePrinter::bvisit(const NaN &x)
    {
        (void)x;
        std::ostringstream s;
        s << "NAN";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const Equality &x)
    {
        std::ostringstream s;
        s << apply(x.get_arg1()) << " == " << apply(x.get_arg2());
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const Unequality &x)
    {
        std::ostringstream s;
        s << apply(x.get_arg1()) << " != " << apply(x.get_arg2());
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const LessThan &x)
    {
        std::ostringstream s;
        s << apply(x.get_arg1()) << " <= " << apply(x.get_arg2());
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const StrictLessThan &x)
    {
        std::ostringstream s;
        s << apply(x.get_arg1()) << " < " << apply(x.get_arg2());
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const UnivariateSeries &x)
    {
        (void)x;
        std::ostringstream s;
        s << "Not supported: ";
        s << __PRETTY_FUNCTION__;
        throw SymEngineException(s.str());
    }

    void CXXCodePrinter::bvisit(const Derivative &x)
    {
        (void)x;
        std::ostringstream s;
        s << "Not supported: ";
        s << __PRETTY_FUNCTION__;
        std::cerr << x << '\n';
        throw SymEngineException(s.str());
    }

    void CXXCodePrinter::bvisit(const Subs &x)
    {
        (void)x;
        std::ostringstream s;
        s << "Not supported: ";
        s << __PRETTY_FUNCTION__;
        throw SymEngineException(s.str());
    }

    void CXXCodePrinter::bvisit(const GaloisField &x)
    {
        (void)x;
        std::ostringstream s;
        s << "Not supported: ";
        s << __PRETTY_FUNCTION__;
        throw SymEngineException(s.str());
    }

    void CXXCodePrinter::bvisit(const Infty &x)
    {
        std::ostringstream s;
        if (x.is_negative_infinity())
            s << "-INFINITY";
        else if (x.is_positive_infinity())
            s << "INFINITY";
        else
            std::ostringstream s;
        s << "Not supported: ";
        s << __PRETTY_FUNCTION__;
        throw SymEngineException(s.str());
        str_ = s.str();
    }

    void CXXCodePrinter::_print_pow(std::ostringstream &o, const RCP<const Basic> &a, const RCP<const Basic> &b)
    {
        if (eq(*a, *E)) {
            o << NAMESPACE_STR "exp(" << apply(b) << ")";
        }
        else if (eq(*b, *rational(1, 2))) {
            o << NAMESPACE_STR "sqrt(" << apply(a) << ")";
        }
        else if (eq(*b, *rational(1, 3))) {
            o << NAMESPACE_STR "cbrt(" << apply(a) << ")";
        }
        else {
            o << NAMESPACE_STR "pow(" << apply(a) << ", " << apply(b) << ")";
        }
    }

    void CXXCodePrinter::bvisit(const Gamma &x)
    {
        std::ostringstream s;
        s << NAMESPACE_STR "tgamma(" << apply(x.get_arg()) << ")";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const LogGamma &x)
    {
        std::ostringstream s;
        s << NAMESPACE_STR "lgamma(" << apply(x.get_arg()) << ")";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const ATan2 &x)
    {
        std::ostringstream s;
        const auto &args = x.get_args();
        switch (args.size()) {
        case 0:
        case 1:
            throw SymEngineException("Impossible");
        case 2:
            s << NAMESPACE_STR "atan2(" << apply(args[0]) << ", " << apply(args[1]) << ")";
            break;
        default: {
            vec_basic inner_args(args.begin() + 1, args.end());
            auto inner = max(inner_args);
            s << NAMESPACE_STR "atan2(" << apply(args[0]) << ", " << apply(inner) << ")";
            break;
        }
        }
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const Log &x)
    {
        std::ostringstream s;
        s << NAMESPACE_STR "log(" << apply(x.get_arg()) << ")";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const Erf &x)
    {
        std::ostringstream s;
        s << NAMESPACE_STR "erf(" << apply(x.get_arg()) << ")";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const Erfc &x)
    {
        std::ostringstream s;
        s << NAMESPACE_STR "erfc(" << apply(x.get_arg()) << ")";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const ASin &x)
    {
        std::ostringstream s;
        s << NAMESPACE_STR "asin(" << apply(x.get_arg()) << ")";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const ACos &x)
    {
        std::ostringstream s;
        s << NAMESPACE_STR "acos(" << apply(x.get_arg()) << ")";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const ATan &x)
    {
        std::ostringstream s;
        s << NAMESPACE_STR "atan(" << apply(x.get_arg()) << ")";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const ACsc &x)
    {
        std::ostringstream s;
        s << NAMESPACE_STR "asin( 1.0 / (" << apply(x.get_arg()) << "))";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const ASec &x)
    {
        std::ostringstream s;
        s << NAMESPACE_STR "acos( 1.0 / (" << apply(x.get_arg()) << "))";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const ACot &x)
    {
        std::ostringstream s;
        s << NAMESPACE_STR "atan( 1.0 / (" << apply(x.get_arg()) << "))";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const ASinh &x)
    {
        std::ostringstream s;
        s << NAMESPACE_STR "asinh(" << apply(x.get_arg()) << ")";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const ACosh &x)
    {
        std::ostringstream s;
        s << NAMESPACE_STR "acosh(" << apply(x.get_arg()) << ")";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const ATanh &x)
    {
        std::ostringstream s;
        s << NAMESPACE_STR "atanh(" << apply(x.get_arg()) << ")";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const ACsch &x)
    {
        std::ostringstream s;
        s << NAMESPACE_STR "asinh( 1.0 / (" << apply(x.get_arg()) << "))";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const ASech &x)
    {
        std::ostringstream s;
        s << NAMESPACE_STR "acosh( 1.0 / (" << apply(x.get_arg()) << "))";
        str_ = s.str();
    }

    void CXXCodePrinter::bvisit(const ACoth &x)
    {
        std::ostringstream s;
        s << NAMESPACE_STR "atanh( 1.0 / (" << apply(x.get_arg()) << "))";
        str_ = s.str();
    }


    std::string cxxcode(const Basic &x)
    {
        CXXCodePrinter c;
        return c.apply(x);
    }

} // namespace SymEngine


std::string applyImplicitMultiplication(const std::string_view &exprOrig)
{
    // trim white space from string_view
    // routine copied from https://terrislinenbach.medium.com/trimming-whitespace-from-a-string-view-6795e18b108f
    auto trim = [](std::string_view in) -> std::string_view {
        auto left = in.begin();
        for (;; ++left) {
            if (left == in.end())
                return std::string_view();
            if (!isspace(*left))
                break;
        }
        auto right = in.end() - 1;
        for (; right > left && isspace(*right); --right)
            ;
        return std::string_view(left, std::distance(left, right) + 1);
    };

    std::string_view expr = trim(exprOrig);

    if (expr.size() <= 1) {
        return std::string(expr.begin(), expr.end());
    }

    // set of standard library functions in <cmath>
    constexpr std::array<std::string_view, 48> known_funcs = { "acos", "acosh", "asin", "asinh", "atan", "atan2", "atanh", "ceil", "copysign", "cos", "cosh", "erf", "erfc", "exp", "expm1", "fabs", "floor", "fmod", "abs", "frexp", "gamma", "hypot", "isclose", "isfinite", "isinf", "isnan", "isqrt", "ldexp", "lgamma", "log", "log1p", "log10", "log2", "modf", "pow", "remainder", "sin", "sinh", "sqrt", "tan", "tanh", "trunc", "sec", "csc", "cot", "sech", "csch", "coth" };

    size_t j, i = 0, k;
    char c;
    size_t nidx = 0;
    size_t idx[expr.size()];

    while (i < expr.size()) {
        c = expr[i];

        if (std::isalpha(c)) {
            j = i + 1;
            // non whitespace
            while ((j < expr.size()) && itemNotInSet(expr[j], ' ', '+', '-', '*', '/', '^', '(', ')')) {
                j++;
            }
            // trailing whitespace
            while ((j < expr.size()) && (expr[j] == ' ')) {
                j++;
            }

            if (j < expr.size()) {
                // check if non whitespace is recognized function
                // if not assume it's a var and apply multiplication
                if (expr[j] == '(') {
                    std::string_view substr = trim(std::string_view(expr.data() + i, j - i));

                    if (std::find(known_funcs.begin(), known_funcs.end(), substr) == known_funcs.end()) {
                        // symbol not in known symbols list -> mark index for *
                        idx[nidx++] = j;
                    }
                }
                // if next non-whitespace is not in set, apply imp mul
                else if (itemNotInSet(expr[j], '+', '-', '*', '/', '^', '(', ')')) {
                    idx[nidx++] = j;
                }
            }
            i = j - 1;
        }
        else if (std::isdigit(c)) {
            j = i + 1;
            // goto end if numeric val
            while ((j < expr.size()) && (std::isdigit(expr[j]) || expr[j] == '.')) {
                j++;
            }
            // if whitespace, go to end of it
            while ((j < expr.size()) && (expr[j] == ' ')) {
                j++;
            }

            // if next non whitespace char not in set, assume imp mul
            if ((j < expr.size()) && itemNotInSet(expr[j], '+', '-', '*', '/', '^', ')')) {
                idx[nidx++] = j;
            }
            i = j - 1;
        }
        else if (c == ')') {
            /*
             * case: ...) [\(0-9a-zA-Z] -> [...) *[\(0-9a-zA-Z]
             */
            j = i + 1;
            while ((j < expr.size()) && (expr[j] == ' ')) {
                j++;
            }
            if ((j < expr.size())) {
                if (std::isalpha(expr[j]) || std::isdigit(expr[j]) || (expr[j] == '(')) {
                    idx[nidx++] = j;
                }
            }
        }
        i++;
    }

    // now that all indeces where * needs to be inserted are known
    // create new string joining each substring by '*' and return the result
    std::string res(expr.size() + nidx + 1lu, 0);
    i = 0;
    k = 0;

    for (size_t jj = 0; jj < nidx; jj++) {
        j = idx[jj];
        std::memcpy(&res[i + k], &expr[i], j - i);
        *(&res[j + k]) = '*';
        k++;
        i = j;
    }
    j = expr.size();
    std::memcpy(&res[i + k], &expr[i], j - i);
    return res;
}
