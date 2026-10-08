#!/usr/bin/env bash
# Run scripts/sync_from_vps.sh inside the Airflow worker with the host user's ssh setup.
# The host's ~/.ssh is mounted read-only at /run/host-ssh. ssh refuses a config or key that others
# can read, and files on a Windows drive show up as mode 0777, so the config, known_hosts and only
# the identity files of the server alias are copied with mode 600 into ~/.ssh, a tmpfs: the key
# never lands on a disk of the container or in the image.
set -euo pipefail
host_ssh=/run/host-ssh
vps=${RADAR_VPS:-radar-vps}

if [ ! -f "$host_ssh/config" ]; then
  echo "$host_ssh/config not found: RADAR_SSH_DIR must point at the ssh directory with the $vps alias" >&2
  exit 1
fi
install -m 600 "$host_ssh/config" ~/.ssh/config
if [ -f "$host_ssh/known_hosts" ]; then
  install -m 600 "$host_ssh/known_hosts" ~/.ssh/known_hosts
fi
ssh -G "$vps" | awk '$1 == "identityfile" { print $2 }' | while read -r key; do
  name=${key##*[/\\]}
  if [ -f "$host_ssh/$name" ]; then
    install -m 600 "$host_ssh/$name" ~/.ssh/"$name"
  fi
done

exec bash /opt/radar/scripts/sync_from_vps.sh "$@"
