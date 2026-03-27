#!/usr/bin/env bash

_twist_completions() {
    local cur prev cmd subcmds opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    cmd="${COMP_WORDS[1]}"

    subcmds="build initialize solve continuation postprocess visualize info tools import simulate local-continuation"

    # 1. Complete the first argument (subcommands)
    if [[ ${COMP_CWORD} -eq 1 ]]; then
        COMPREPLY=( $(compgen -W "${subcmds}" -- "${cur}") )
        return 0
    fi

    # 2. Complete options based on the specific subcommand
    case "${cmd}" in
        build)
            opts="-c --compiler -s --source-only"
            ;;
        initialize)
            opts="-p --parameter-set --time-ode --time-pde --stim-time --stim-dur --stim-amp --num-space -m --method -d --display-start-and-end -r --refine-rest-state -o --stop-at-ode"
            ;;
        solve)
            opts="--parameter-set --from-data --solve-init --ncol --nthreads --solve-tol --solve-geps --solve-min-damp --solve-max-iter --solve-nadapt --solve-min-nodes --solve-max-nodes -q --solve-quietly --plot-after-solve-init --plot-dense --spectrum --essential-spectrum --essential-spectrum-filter --spectrum-strategy --subspace --subspace-size --subspace-sigma --simulate --simulate-adaptive-mesh --simulate-nspace --simulate-nsample --simulate-nloops --simulate-xeps --simulate-animate --simulate-method --simulate-save-path --add-bumps --simulate-vscale --replace-existing-initial-guess"
            ;;
        continuation)
            opts="--parameter-set --from-data -f --forward -b --backward -p --parameter --ds --dsmin --dsmax --parmin --parmax --solve-init --ncol --nthreads --solve-tol --solve-geps --solve-min-damp --solve-max-iter --solve-nadapt --solve-min-nodes --solve-max-nodes -q --solve-quietly -t --tag --prefix --add-bumps"
            ;;
        postprocess)
            opts="-s --spectrum -w --which -b --bifurcations -l --length --spectrum-strategy --less-memory"
            ;;
        visualize)
            opts="--l2 --apd --individual --no-stability --actual-spatial-period --center-wave --hide-bifurcation-points --print-rc-file --dashed-unstable --apply-transforms --apd-value --dispersion-mode --rc-file --save --spatial-period-units --wave-speed-units --wave-profile-units --dispersion-labels --fixed-solutions --starting-extents --stable-color --unstable-color"
            ;;
        info)
            opts="-l --length -s --spectrum -i --indices -p --show-params -m --print-meta-data -w --which-params"
            ;;
        tools)
            opts="-i --indices -p --prune -r --reverse -a --append -n --insert -f --flip-direction -x --export-to -s --solution-to-export -o --export-path --import-initial-wave --refine-between --refine-ds"
            ;;
        import)
            opts="-p --parameter-set -f --file-type -c --wave-speed --derivatives-included"
            ;;
        simulate)
            opts="--adaptive-mesh --nspace --nsample --nloops --xeps --animate --method --save-data --vscale --colorbar --apply-transforms --hide-ticks --rc-file --colorbar-label --xlabel --ylabel --save-fig"
            ;;
        local-continuation)
            opts="--parameter-set --from-data -f --forward -b --backward -p --parameter --wave-speed --ds --dsmin --dsmax --parmin --parmax -t --tag --prefix"
            ;;
        *)
            opts=""
            ;;
    esac

    # 3. Restrict completion strictly to the defined flags/options
    if [[ ${cur} == -* ]] || [[ -z ${cur} ]]; then
        COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
    fi

    return 0
}

# Register the completion function (with default file completion disabled)
complete -F _twist_completions twist
