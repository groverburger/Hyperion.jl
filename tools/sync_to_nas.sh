#!/usr/bin/env bash
# Copy Hyperion.jl's tracked files to a NAS checkout, without git history.
#
# Untracked files on NAS (data, logs, Manifest.toml, PBS scripts) are left
# alone. A file that an earlier sync copied and that is no longer tracked
# is removed, so renames and deletions carry over.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: tools/sync_to_nas.sh --host=HOST [options]

  --host=HOST       NAS front end to copy to
                    (default: $NAS_HOST or zbooth@cfe01)
  --jump=HOST       secure front end to pass through with NAS ssh-proxy;
                    empty for a direct connection
                    (default: $NAS_JUMP or zbooth@sfe6.nas.nasa.gov)
  --dest=PATH       repository directory on NAS
                    (default: $NAS_DEST or /nobackup/zbooth/Hyperion.jl)
  --working-tree    send tracked files as they are on disk, including
                    uncommitted changes (default: the committed HEAD)
  --dry-run         show what would be sent without connecting
  -h, --help        show this help

Both hops authenticate from this terminal, so expect the sfe password and
passcode prompts, then the cfe passcode prompt. Set NAS_SSH to replace the
ssh program.
EOF
}

host="${NAS_HOST:-zbooth@cfe01}"
jump="${NAS_JUMP-zbooth@sfe6.nas.nasa.gov}"
dest="${NAS_DEST:-/nobackup/zbooth/Hyperion.jl}"
working_tree=0
dry_run=0

for arg in "$@"; do
    case "$arg" in
        --host=*) host="${arg#*=}" ;;
        --dest=*) dest="${arg#*=}" ;;
        --jump=*) jump="${arg#*=}" ;;
        --working-tree) working_tree=1 ;;
        --dry-run) dry_run=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
    esac
done

[ -n "$host" ] || { echo "Missing --host (or NAS_HOST)" >&2; exit 2; }
# The remote command is run by the login shell, which may be csh, so the
# path is single-quoted and must not contain quotes or history characters.
case "$dest" in
    /*) ;;
    *) echo "--dest must be an absolute path" >&2; exit 2 ;;
esac
case "$dest" in
    *\'*|*\!*|*\"*) echo "--dest must not contain quotes or '!'" >&2; exit 2 ;;
esac

repo="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$repo"

revision="$(git rev-parse --short HEAD)"
dirty="$(git status --porcelain --untracked-files=no | wc -l | tr -d ' ')"

staging="$(mktemp -d "${TMPDIR:-/tmp}/hyperion_sync.XXXXXX")"
trap 'rm -rf "$staging"' EXIT

# Stage exactly the files to send, so the manifest matches the archive.
export COPYFILE_DISABLE=1   # no macOS ._ files in the archive
if [ "$working_tree" -eq 1 ]; then
    source_label="working tree at $revision ($dirty modified tracked files)"
    git ls-files -z | while IFS= read -r -d '' f; do
        [ -e "$f" ] && printf '%s\0' "$f"   # skip tracked files deleted on disk
    done | tar --null -T - --no-xattrs -cf - | tar -xf - -C "$staging"
else
    source_label="HEAD $revision"
    [ "$dirty" -gt 0 ] &&
        echo "Note: $dirty tracked files have uncommitted changes that are not sent (see --working-tree)." >&2
    git archive --format=tar HEAD | tar -xf - -C "$staging"
fi

meta="$staging/.hyperion_sync"
mkdir -p "$meta"
(cd "$staging" && find . -path ./.hyperion_sync -prune -o \( -type f -o -type l \) -print |
    sed 's|^\./||' | LC_ALL=C sort) > "$meta/manifest.new"
printf 'source: %s\nhost: %s\nsynced_at: %s\n' \
    "$source_label" "$(hostname)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$meta/info.new"

# Run on NAS after extraction: delete files that the previous sync sent and
# that are no longer tracked, then record this sync's manifest.
cat > "$meta/apply.sh" <<'EOF'
set -eu
cd "$(dirname "$0")/.."
if [ -f .hyperion_sync/manifest ]; then
    LC_ALL=C comm -23 .hyperion_sync/manifest .hyperion_sync/manifest.new |
    while IFS= read -r f; do
        if [ -f "$f" ] || [ -L "$f" ]; then
            rm -f -- "$f" && echo "removed: $f"
        fi
    done
fi
mv .hyperion_sync/manifest.new .hyperion_sync/manifest
mv .hyperion_sync/info.new .hyperion_sync/info
echo "NAS checkout updated:"
sed 's/^/  /' .hyperion_sync/info
EOF

count="$(wc -l < "$meta/manifest.new" | tr -d ' ')"
size="$(du -sh "$staging" | cut -f1)"
echo "Source:      $source_label"
echo "Destination: $host:$dest${jump:+ (via $jump)}"
echo "Files:       $count ($size before compression)"

if [ "$dry_run" -eq 1 ]; then
    echo "Dry run: nothing sent."
    exit 0
fi

# One connection through the secure front end, so each hop authenticates
# once. The SFE does not forward by hostname (ssh -J fails to resolve NAS
# hosts); its ssh-proxy command connects to the host instead. Prompts are
# read from the terminal, not from the archive on stdin.
ssh_cmd=(${NAS_SSH:-ssh} -o ForwardX11=no)
[ -n "$jump" ] &&
    ssh_cmd+=(-o "ProxyCommand=${NAS_SSH:-ssh} -o ForwardX11=no $jump ssh-proxy %h")
# Archive the staged entries by name, not as ".", so extraction does not
# change the mode of the existing destination directory (GNU tar would).
entries=()
while IFS= read -r -d '' entry; do entries+=("${entry#./}"); done \
    < <(cd "$staging" && find . -mindepth 1 -maxdepth 1 -print0)
tar -C "$staging" --no-xattrs -czf - "${entries[@]}" |
    "${ssh_cmd[@]}" "$host" \
        "mkdir -p '$dest' && cd '$dest' && tar -xzmf - --no-same-owner && sh .hyperion_sync/apply.sh"
