#!/usr/bin/env bash
# Write a PBS job for a mapset spec, copy both to NAS, and submit the job.
#
# The local spec is copied next to the PBS file, so the job runs exactly the
# spec on this machine. Layer paths in it still start at the NAS repository
# root, and its sha256 fields are checked by generate_mapset.jl on NAS.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: tools/launch_on_nas.sh --spec=PATH [options] [-- generate_mapset options]

  --spec=PATH       local mapset TOML to run
  --gpus=N          GPUs on one node (default 4)
  --ncpus=N         CPU cores (default 16 per GPU)
  --mem=SIZE        memory (default 64GB per GPU)
  --walltime=H:M:S  wall time (default 6:00:00)
  --queue=QUEUE     PBS queue (default gpu_normal@pbspl4)
  --model=MODEL     node model; NAS GPU queues require one (default mil_a100)
  --check           run generate_mapset.jl --dry-run on the front end first;
                    stops before qsub if paths or hashes are wrong
  --no-submit       copy the files but do not run qsub
  --dry-run         print the PBS file without connecting
  -h, --help        show this help

Options after -- go to generate_mapset.jl, e.g.
  -- --start=2027-09-14T00:00:00 --stop=2027-10-22T23:00:00 --overwrite

Connection settings are the same as tools/sync_to_nas.sh: NAS_HOST,
NAS_JUMP, and NAS_SSH. NAS_ROOT (default /nobackup/zbooth) is where the PBS
file goes; the repository is $NAS_ROOT/Hyperion.jl. NAS_ENV is the shell
file that sets up Julia and CUDA (default $NAS_ROOT/env/julia_cuda_env.sh).
EOF
}

host="${NAS_HOST:-zbooth@cfe01}"
jump="${NAS_JUMP-zbooth@sfe6.nas.nasa.gov}"
root="${NAS_ROOT:-/nobackup/zbooth}"
env_file="${NAS_ENV:-$root/env/julia_cuda_env.sh}"
spec=""
gpus=4
ncpus=""
mem=""
walltime="6:00:00"
queue="gpu_normal@pbspl4"
model="mil_a100"
check=0
submit=1
dry_run=0
mapset_args=()

while [ $# -gt 0 ]; do
    case "$1" in
        --spec=*) spec="${1#*=}" ;;
        --gpus=*) gpus="${1#*=}" ;;
        --ncpus=*) ncpus="${1#*=}" ;;
        --mem=*) mem="${1#*=}" ;;
        --walltime=*) walltime="${1#*=}" ;;
        --queue=*) queue="${1#*=}" ;;
        --model=*) model="${1#*=}" ;;
        --check) check=1 ;;
        --no-submit) submit=0 ;;
        --dry-run) dry_run=1 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; mapset_args=("$@"); break ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

[ -n "$spec" ] || { echo "Missing --spec" >&2; exit 2; }
[ -f "$spec" ] || { echo "Spec not found: $spec" >&2; exit 2; }
[[ "$gpus" =~ ^[1-9][0-9]*$ ]] || { echo "--gpus must be a positive integer" >&2; exit 2; }
ncpus="${ncpus:-$((16 * gpus))}"
mem="${mem:-$((64 * gpus))GB}"
# Paths are single-quoted in the remote command, which may run under csh.
for p in "$root" "$env_file"; do
    case "$p" in
        /*) ;;
        *) echo "NAS paths must be absolute: $p" >&2; exit 2 ;;
    esac
    case "$p" in
        *\'*|*\!*|*\"*|*' '*) echo "NAS paths must not contain quotes, spaces or '!': $p" >&2; exit 2 ;;
    esac
done

# One name for the PBS file, spec copy, and job log, unique per launch.
stem="$(basename "$spec" .toml | tr -c 'A-Za-z0-9_.\n-' '_')"
base="hyperion_${stem}_$(date +%Y%m%d-%H%M%S)"
job_name="$(printf 'h_%s' "$stem" | cut -c1-15)"
remote_spec="$root/$base.toml"
remote_pbs="$root/$base.pbs"
remote_log="$root/$base.log"

[ -n "$model" ] || { echo "--model must not be empty; NAS GPU queues require it" >&2; exit 2; }
select="select=1:ncpus=$ncpus:ngpus=$gpus:mem=$mem:model=$model"

command=(julia --project=. scripts/generate_mapset.jl "--spec=$remote_spec"
         --backend=cuda "--gpus=$gpus" ${mapset_args[@]+"${mapset_args[@]}"})  # bash 3.2-safe empty array
revision="$(git -C "$(dirname "$0")" rev-parse --short HEAD 2>/dev/null || echo unknown)"

staging="$(mktemp -d "${TMPDIR:-/tmp}/hyperion_launch.XXXXXX")"
trap 'rm -rf "$staging"' EXIT
cp "$spec" "$staging/$base.toml"

cat > "$staging/$base.pbs" <<EOF
#PBS -N $job_name
#PBS -q $queue
#PBS -l $select
#PBS -l walltime=$walltime
#PBS -S /bin/bash
#PBS -j oe
#PBS -o $remote_log
# Written by tools/launch_on_nas.sh on $(hostname) at $(date -u +%Y-%m-%dT%H:%M:%SZ)
# from $spec (local Hyperion.jl $revision).

cd $root/Hyperion.jl
source $env_file

hostname
date
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES"
if [ -f .hyperion_sync/info ]; then cat .hyperion_sync/info; fi

export JULIA_NUM_THREADS=$ncpus
julia --project=. -e 'using CUDA; CUDA.versioninfo(); @assert CUDA.functional()' || {
    echo "CUDA is not functional on this node; stopping."
    exit 1
}

$(printf '%q ' "${command[@]}")
status=\$?

date
echo "generate_mapset.jl finished with status \$status"
exit \$status
EOF

echo "Spec:        $spec"
echo "Destination: $host:$root${jump:+ (via $jump)}"
echo "Job:         $job_name, $select, walltime $walltime, queue $queue"
echo "PBS file:    $remote_pbs"
echo "Job log:     $remote_log"

if [ "$dry_run" -eq 1 ]; then
    echo
    cat "$staging/$base.pbs"
    echo
    echo "Dry run: nothing sent."
    exit 0
fi

# The remote step runs under bash, whatever the login shell, and reports
# what it did. qsub is attempted only when the earlier steps succeed.
cat > "$staging/$base.launch.sh" <<EOF
set -e
cd '$root'
rm -f '$base.launch.sh'
if [ $check -eq 1 ]; then
    echo "Checking the spec with generate_mapset.jl --dry-run..."
    (cd Hyperion.jl && source '$env_file' >/dev/null &&
        julia --project=. scripts/generate_mapset.jl --spec='$remote_spec' --dry-run) || {
        echo "Check failed; the job was not submitted. Files are in $root." >&2
        exit 1
    }
fi
if [ $submit -eq 0 ]; then
    echo "Not submitted. To submit: qsub $remote_pbs"
    exit 0
fi
if ! command -v qsub >/dev/null 2>&1; then
    echo "qsub is not available on \$(hostname). To submit: qsub $remote_pbs" >&2
    exit 1
fi
if job=\$(qsub '$remote_pbs'); then
    echo "Submitted \$job"
    echo "Status:  qstat -u \$USER"
    echo "Log:     $remote_log (when the job runs)"
else
    echo "qsub failed. To submit: qsub $remote_pbs" >&2
    exit 1
fi
EOF

ssh_cmd=(${NAS_SSH:-ssh} -o ForwardX11=no)
[ -n "$jump" ] &&
    ssh_cmd+=(-o "ProxyCommand=${NAS_SSH:-ssh} -o ForwardX11=no $jump ssh-proxy %h")
COPYFILE_DISABLE=1 tar -C "$staging" --no-xattrs -cf - \
    "$base.toml" "$base.pbs" "$base.launch.sh" |
    "${ssh_cmd[@]}" "$host" \
        "mkdir -p '$root' && cd '$root' && tar -xf - && bash '$base.launch.sh'"
