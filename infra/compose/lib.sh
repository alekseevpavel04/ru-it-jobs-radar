# Sourced by scripts/up.sh and scripts/down.sh.
#   RADAR_ENV_FILE    env file with the stack variables, default .env at the repo root
#   RADAR_WSL_DISTRO  WSL distro with Docker Engine, default radar (used from Git Bash on Windows)
#   RADAR_SSH_DIR     ssh directory with the collection server's alias, mounted read-only into the
#                     Airflow worker; default ~/.ssh of the user who runs the script (from Git Bash
#                     the Windows user's, not the WSL user's)
#   AIRFLOW_UID       uid the Airflow services run as, default the uid running docker (owner of the
#                     bind-mounted files)

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
    --exec env RADAR_ENV_FILE="$(to_wsl_path "$env_file")" RADAR_SSH_DIR="$(to_wsl_path "${RADAR_SSH_DIR:-$HOME/.ssh}")" \
    bash "scripts/$script" "$@"
}

require_env_file() {
  RADAR_ENV_FILE=${RADAR_ENV_FILE:-$RADAR_ROOT/.env}
  if [ ! -f "$RADAR_ENV_FILE" ]; then
    echo "env file $RADAR_ENV_FILE not found: copy .env.example to .env and fill it in" >&2
    exit 1
  fi
}

compose() {
  AIRFLOW_UID=${AIRFLOW_UID:-$(id -u)} RADAR_SSH_DIR=${RADAR_SSH_DIR:-$HOME/.ssh} docker compose --file "$RADAR_COMPOSE_FILE" --env-file "$RADAR_ENV_FILE" "$@"
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

# Secrets that only one profile needs. Compose interpolates the whole file for every profile, so a
# `:?` on them would stop `up.sh core` as well; they are checked here for the profiles being started.
declare -A PROFILE_SECRETS=(
  [orchestration]="AIRFLOW_DB_PASSWORD AIRFLOW_ADMIN_PASSWORD AIRFLOW_JWT_SECRET"
  [mlops]="N8N_OWNER_PASSWORD"
)

# Value of a variable: the environment first, then the last assignment in the env file.
env_value() {
  local name=$1 value=${!1:-}
  if [ -z "$value" ]; then
    value=$(sed -n "s/^[[:space:]]*$name=//p" "$RADAR_ENV_FILE" | tail -n 1 | tr -d "\"'[:space:]")
  fi
  printf '%s' "$value"
}

# Exits naming every secret of the selected profiles (PROFILE_ARGS) that is empty both in the
# environment and in the env file.
require_profile_secrets() {
  local i name missing=()
  for ((i = 1; i < ${#PROFILE_ARGS[@]}; i += 2)); do
    for name in ${PROFILE_SECRETS[${PROFILE_ARGS[i]}]:-}; do
      [ -n "$(env_value "$name")" ] || [[ " ${missing[*]} " == *" $name "* ]] || missing+=("$name")
    done
  done
  if [ ${#missing[@]} -gt 0 ]; then
    echo "set ${missing[*]} in $RADAR_ENV_FILE (see .env.example)" >&2
    exit 1
  fi
}
