#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <folder_path> [--no-albedo | --lone | --scale-albedo | --maxiter <num_iter> | --res <resolution> | --no-opti-lights | --no-rgbplus | --snapshot <snapshot_path> | --supernormal | --export-network | --save-mesh]"
    exit 1
fi

case="$1"

cp run.sh "$case/run.sh.log"
echo "./run.sh $@" > "$case/command.log"

flags=""
scale_albedo=false
export_network=false
save_mesh=false

while [ $# -gt 1 ]; do
    case "$2" in
        --no-albedo)
            flags="$flags --no-albedo"
            ;;
        --lone)
            flags="$flags --ltwo"
            ;;
        --scale-albedo)
            scale_albedo=true
            ;;
        --maxiter)
            num_iter="$3"
            flags="$flags --maxiter $num_iter"
            shift
            ;;
        --res)
            resolution="$3"
            flags="$flags --res $resolution"
            shift
            ;;
        --no-opti-lights)
            flags="$flags --no-opti-lights"
            no_opti_lights=true
            ;;
        --no-rgbplus)
            flags="$flags --no-rgbplus"
            ;;
        --snapshot)
            flags="$flags --snapshot $3"
            shift
            ;;
        --supernormal)
            flags="$flags --supernormal"
            supernormal=true
            ;;
        --export-network)
            export_network=true
            ;;
        --save-mesh)
            save_mesh=true
            ;;
        *)
            echo "Unknown option: $2"
            exit 1
            ;;
    esac
    shift
done

num_iter=${num_iter:-15000}
resolution=${resolution:-1024}
iter_opti_lights=${iter_opti_lights:-$(($num_iter/3*2))}
supernormal=${supernormal:-false}

# Build the final-step extra flags
final_flags=""
if [ "$export_network" = true ]; then
    final_flags="$final_flags --export-network"
fi
if [ "$save_mesh" = true ]; then
    final_flags="$final_flags --save-mesh"
fi

if [ "$scale_albedo" = true ]; then
    flags=$(echo "$flags" | sed 's/--no-albedo//')
    ./run.sh "$case" --no-albedo $flags
    python scripts/scale_albedos.py --folder "$case"
    path=$(dirname "$case")
    folder=$(basename "$case")
    folder="${folder}-albedoscaled"
    ./run.sh "$path/$folder/" $flags
    exit 0
fi

flags=$(echo "$flags" | sed 's/--maxiter [0-9]*//')
flags=$(echo "$flags" | sed 's/--no-opti-lights//')
flags=$(echo "$flags" | sed 's/--res [0-9]*//')
flags=$(echo "$flags" | sed 's/--supernormal//')

if [ "$supernormal" = false ]; then

    echo "./build/testbed --scene ${case}/ --maxiter ${iter_opti_lights} --save-snapshot --mask-weight 1.0 --no-gui $flags"
    ./build/testbed --scene "${case}/" --maxiter "${iter_opti_lights}" --save-snapshot --mask-weight 1.0 --no-gui $flags

    flags=$(echo "$flags" | sed 's/--snapshot [^ ]*//')

    if [ "$no_opti_lights" = true ]; then
        echo "./build/testbed --scene ${case}/ --maxiter ${num_iter} --save-snapshot --mask-weight 1.0 --no-gui --snapshot ${case}/snapshot_${iter_opti_lights}.msgpack --save-mesh --resolution ${resolution} --opti-lights $flags $final_flags"
        ./build/testbed --scene "${case}/" --maxiter "${num_iter}" --save-snapshot --mask-weight 1.0 --no-gui --snapshot "${case}/snapshot_${iter_opti_lights}.msgpack" --save-mesh --resolution "${resolution}" $flags $final_flags
        exit 0
    fi

    echo "./build/testbed --scene ${case}/ --maxiter ${num_iter} --save-snapshot --mask-weight 1.0 --no-gui --snapshot ${case}/snapshot_${iter_opti_lights}.msgpack --save-mesh --resolution ${resolution} --opti-lights $flags $final_flags"
    ./build/testbed --scene "${case}/" --maxiter "${num_iter}" --save-snapshot --mask-weight 1.0 --no-gui --snapshot "${case}/snapshot_${iter_opti_lights}.msgpack" --save-mesh --resolution "${resolution}" --opti-lights $flags $final_flags
    exit 0

elif [ "$supernormal" = true ]; then

    echo "./build/testbed --scene ${case}/ --maxiter ${num_iter} --save-snapshot --save-mesh --mask-weight 1.0 --no-gui --resolution ${resolution} --supernormal $flags $final_flags"
    ./build/testbed --scene "${case}/" --maxiter "${num_iter}" --save-snapshot --save-mesh --mask-weight 1.0 --no-gui --resolution ${resolution} --supernormal $flags $final_flags
    exit 0

fi