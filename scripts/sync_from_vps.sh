#!/usr/bin/env bash
# Pull complete raw partitions collected on the daily server into the local raw layer (docs/vps.md).
# It only adds: a local part file is never replaced and nothing local is deleted. A partition is
# pulled when the local copy is absent or holds a subset of the server's parts (the server started
# from a copy of it); then the missing parts land first and the `_SUCCESS` marker last. A day both
# machines collected on their own is left alone and reported, with its run log lines.
# `_meta/*.jsonl` and `_rejects/*.jsonl` get the server's lines that are not yet local, other
# `_meta` files are copied when absent.
#   scripts/sync_from_vps.sh                 trudvsem habr trudvsem_snapshot
#   scripts/sync_from_vps.sh habr            only these
# RADAR_VPS       ssh destination, an alias from ~/.ssh/config (default radar-vps)
# RADAR_VPS_DIR   project directory on the server, relative to its home (default ru-it-jobs-radar)
# RADAR_DATA_DIR  local data directory (default data)
set -euo pipefail
cd "$(dirname "$0")/.."

vps=${RADAR_VPS:-radar-vps}
remote_dir=${RADAR_VPS_DIR:-ru-it-jobs-radar}
raw=${RADAR_DATA_DIR:-data}/raw
sources=("$@")
[ ${#sources[@]} -gt 0 ] || sources=(trudvsem habr trudvsem_snapshot)
mkdir -p "$raw"

# One line per file: S marker md5 / P part size / M metadata file, all relative to data/raw.
listing=$(
  ssh -o BatchMode=yes "$vps" bash -s -- "$remote_dir" "${sources[@]}" <<'EOF'
set -euo pipefail
cd "$1/data/raw"
shift
for s in "$@"; do
  [ -d "$s" ] || continue
  for d in "$s"/dt=*; do
    [ -f "$d/_SUCCESS" ] || continue
    printf 'S\t%s\t%s\n' "$d" "$(md5sum <"$d/_SUCCESS" | cut -c1-32)"
    for p in "$d"/part-*.parquet; do
      if [ -f "$p" ]; then printf 'P\t%s\t%s\n' "$p" "$(stat -c %s "$p")"; fi
    done
  done
  for f in "$s"/_meta/* "$s"/_rejects/*; do
    if [ -f "$f" ]; then printf 'M\t%s\t-\n' "$f"; fi
  done
done
EOF
)

declare -A marker parts skipped_days
partitions=()
metas=()
while IFS=$'\t' read -r kind path info; do
  case $kind in
    S)
      partitions+=("$path")
      marker[$path]=$info
      ;;
    P) parts[${path%/*}]+=" ${path##*/}:$info" ;;
    M) metas+=("$path") ;;
  esac
done <<<"$listing"

pull=()
fresh=()
for d in "${partitions[@]}"; do
  missing=()
  diverged=
  for entry in ${parts[$d]:-}; do
    name=${entry%:*}
    if [ ! -e "$raw/$d/$name" ]; then
      missing+=("$d/$name")
    elif [ "$(stat -c %s "$raw/$d/$name")" != "${entry##*:}" ]; then
      diverged="$name differs in size"
    fi
  done
  for p in "$raw/$d"/part-*.parquet; do
    [ -e "$p" ] || continue
    case "${parts[$d]:-} " in
      *" ${p##*/}:"*) ;;
      *) diverged="local ${p##*/} is not on the server" ;;
    esac
  done
  if [ -n "$diverged" ]; then
    echo "skip $d: both machines collected it ($diverged)"
    skipped_days[${d%%/*}]+="|${d##*dt=}"
    continue
  fi
  local_marker=
  [ ! -f "$raw/$d/_SUCCESS" ] || local_marker=$(md5sum <"$raw/$d/_SUCCESS" | cut -c1-32)
  [ ${#missing[@]} -gt 0 ] || [ "$local_marker" != "${marker[$d]}" ] || continue
  pull+=("${missing[@]}" "$d/_SUCCESS")
  fresh+=("$d")
done

declare -A fresh_set
for d in "${fresh[@]}"; do fresh_set[$d]=1; done
for f in "${metas[@]}"; do
  name=${f##*/}
  case $f in
    */_rejects/*)
      if [ -n "${fresh_set[${f%%/*}/${name%.jsonl}]:-}" ]; then pull+=("$f"); fi
      ;;
    *.jsonl) pull+=("$f") ;;
    *)
      if [[ $name =~ dt=([0-9-]{10}) ]] && [[ "${skipped_days[${f%%/*}]:-}|" == *"|${BASH_REMATCH[1]}|"* ]]; then
        continue
      fi
      [ -e "$raw/$f" ] || pull+=("$f")
      ;;
  esac
done

if [ ${#fresh[@]} -eq 0 ]; then
  echo "no new or grown partitions on $vps"
fi
[ ${#pull[@]} -gt 0 ] || exit 0

staging=$(mktemp -d "$raw/.sync.XXXXXX")
trap 'rm -rf "$staging"' EXIT
printf '%s\n' "${pull[@]}" |
  ssh -o BatchMode=yes "$vps" "cd '$remote_dir/data/raw' && tar cf - --files-from=-" |
  tar xf - -C "$staging"

for d in "${fresh[@]}"; do
  mkdir -p "$raw/$d"
  n=0
  for p in "$staging/$d"/part-*.parquet; do
    [ -e "$p" ] || continue
    mv -n "$p" "$raw/$d/"
    if [ -e "$p" ]; then
      echo "$raw/$d/${p##*/} appeared locally during the sync; stopped before the marker" >&2
      exit 1
    fi
    n=$((n + 1))
  done
  mv -f "$staging/$d/_SUCCESS" "$raw/$d/_SUCCESS"
  rows=$(grep -o '"rows_in_partition": [0-9]*' "$raw/$d/_SUCCESS" | grep -o '[0-9]*$' || echo '?')
  echo "pulled $d: $n new parts, $rows rows in the partition"
done

# Run log lines of a skipped day describe parts that are not here, so they stay on the server.
for f in "${metas[@]}"; do
  staged=$staging/$f
  [ -f "$staged" ] || continue
  target=$raw/$f
  mkdir -p "${target%/*}"
  case $f in
    *.jsonl)
      skip=
      for day in $(tr '|' ' ' <<<"${skipped_days[${f%%/*}]:-}"); do
        skip+="${skip:+|}\"dt\": \"$day\"|\"version\": \"${day//-/}T"
      done
      [ -f "$target" ] || : >"$target"
      # FILENAME, not NR == FNR: the local file may be empty. Lines written on Windows end in CRLF.
      awk -v skip="$skip" '{ sub(/\r$/, "") }
        FILENAME == ARGV[1] { seen[$0] = 1; next }
        !($0 in seen) && (skip == "" || $0 !~ skip) { print; seen[$0] = 1 }' "$target" "$staged" >"$staged.new"
      added=$(wc -l <"$staged.new")
      cat "$staged.new" >>"$target"
      [ "$added" -eq 0 ] || echo "appended $f: $added lines"
      ;;
    *)
      mv -n "$staged" "$target"
      echo "copied $f"
      ;;
  esac
done
