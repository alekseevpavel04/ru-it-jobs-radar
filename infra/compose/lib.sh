# Sourced by scripts/up.sh and scripts/down.sh.
#   RADAR_ENV_FILE    env file with the stack variables, default .env at the repo root
#   RADAR_WSL_DISTRO  WSL distro with Docker Engine, default radar (used from Git Bash on Windows)

RADAR_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RADAR_COMPOSE_FILE=$RADAR_ROOT/infra/compose/docker-compose.yml

# D:/a/b, /d/a/b or a relative path -> /mnt/d/a/b
to_wsl_path() {
  local p drive
  p=$(cygpath -am "$1")
  drive=$(tr '[:upper:]' '[:lower:]' <<<"${p:0:1}")
  printf '/mnt/%s%s' "$drive" "${p:2}"
}

# Docker lives inside the WSL distro, so from Git Bash the calling script re-runs itself there.
# --exec skips the distro's shell, which would otherwise expand the arguments a second time.
reexec_in_wsl() {
  local script=$1
  shift
  case "$(uname -s)" in
    MINGW* | MSYS* | CYGWIN*) ;;
    *) return 0 ;;
  esac
  local env_file=${RADAR_ENV_FILE:-$RADAR_ROOT/.env}
  MSYS_NO_PATHCONV=1 exec wsl.exe -d "${RADAR_WSL_DISTRO:-radar}" --cd "$(to_wsl_path "$RADAR_ROOT")" \
    --exec env RADAR_ENV_FILE="$(to_wsl_path "$env_file")" bash "scripts/$script" "$@"
}

require_env_file() {
  RADAR_ENV_FILE=${RADAR_ENV_FILE:-$RADAR_ROOT/.env}
  if [ ! -f "$RADAR_ENV_FILE" ]; then
    echo "env file $RADAR_ENV_FILE not found: copy .env.example to .env and fill it in" >&2
    exit 1
  fi
}

compose() {
  docker compose --file "$RADAR_COMPOSE_FILE" --env-file "$RADAR_ENV_FILE" "$@"
}

# Fills the PROFILE_ARGS array with --profile flags: the given profiles, or every profile in the file.
select_profiles() {
  local known p
  known=$(compose config --profiles)
  if [ $# -eq 0 ]; then
    set -- $known
  fi
  PROFILE_ARGS=()
  for p in "$@"; do
    if ! grep -qx -- "$p" <<<"$known"; then
      echo "unknown profile '$p'; defined profiles: $(tr '\n' ' ' <<<"$known")" >&2
      exit 1
    fi
    PROFILE_ARGS+=(--profile "$p")
  done
}
