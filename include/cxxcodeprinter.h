#ifndef CXXCODEPRINTER_H
#define CXXCODEPRINTER_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include <string_view>
#include <symengine/expression.h>
#include <symengine/parser.h>
#include <symengine/printers/codegen.h>

#pragma GCC diagnostic pop

#define NAMESPACE_STR "std::"

namespace SymEngine
{

    class CXXCodePrinter : public BaseVisitor<CXXCodePrinter, StrPrinter>
    {
    public:
        using StrPrinter::apply;
        using StrPrinter::bvisit;
        using StrPrinter::str_;
        void bvisit(const Basic &x);
        void bvisit(const Complex &x);
        void bvisit(const Interval &x);
        void bvisit(const Contains &x);
        void bvisit(const Piecewise &x);
        void bvisit(const Rational &x);
        void bvisit(const EmptySet &x);
        void bvisit(const FiniteSet &x);
        void bvisit(const Reals &x);
        void bvisit(const Rationals &x);
        void bvisit(const Integers &x);
        void bvisit(const UniversalSet &x);
        void bvisit(const Constant &x);
        void bvisit(const NaN &x);
        void bvisit(const Equality &x);
        void bvisit(const Unequality &x);
        void bvisit(const LessThan &x);
        void bvisit(const StrictLessThan &x);
        void bvisit(const UnivariateSeries &x);
        void bvisit(const Derivative &x);
        void bvisit(const Subs &x);
        void bvisit(const GaloisField &x);
        void bvisit(const Infty &x);

        void bvisit(const Abs &x);

        // trig
        void bvisit(const Sin &x);
        void bvisit(const Cos &x);
        void bvisit(const Tan &x);
        void bvisit(const Csc &x);
        void bvisit(const Sec &x);
        void bvisit(const Cot &x);
        // inverse trig
        void bvisit(const ASin &x);
        void bvisit(const ACos &x);
        void bvisit(const ATan &x);
        void bvisit(const ACsc &x);
        void bvisit(const ASec &x);
        void bvisit(const ACot &x);
        void bvisit(const ATan2 &x);

        // hyperbolic
        void bvisit(const Sinh &x);
        void bvisit(const Cosh &x);
        void bvisit(const Tanh &x);
        void bvisit(const Csch &x);
        void bvisit(const Sech &x);
        void bvisit(const Coth &x);
        // inverse hyperbolic
        void bvisit(const ASinh &x);
        void bvisit(const ACosh &x);
        void bvisit(const ATanh &x);
        void bvisit(const ACsch &x);
        void bvisit(const ASech &x);
        void bvisit(const ACoth &x);

        void bvisit(const Log &x);
        void bvisit(const Erf &x);
        void bvisit(const Erfc &x);
        void bvisit(const Pow &x);
        void bvisit(const Gamma &x);
        void bvisit(const LogGamma &x);
        void bvisit(const Floor &x);
        void bvisit(const Ceiling &x);
        void bvisit(const Truncate &x);

        void bvisit(const Max &x);
        void bvisit(const Min &x);

        void _print_pow(std::ostringstream &o, const RCP<const Basic> &a, const RCP<const Basic> &b);
    };

    std::string cxxcode(const Basic &x);

} // namespace SymEngine

template <typename X, typename... Ts>
bool itemNotInSet(const X &x, const Ts &...rest)
{
    return ((x != rest) && ...);
}

std::string applyImplicitMultiplication(const std::string_view &expr);

#endif // CXXCODEPRINTER_H