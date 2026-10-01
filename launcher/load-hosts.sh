# shellcheck shell=bash
# Sourced by vllm-rank.sh, ring-up.sh and the ring-wide bench / kernel-tuning scripts: reads the site facts from
# hosts.env (HOSTS_ENV overrides the path). Plain KEY=VALUE lines, '#' comments, values may use $HOME and quotes.
# A variable that is already set in the environment wins over the file, so one-off overrides still work.
HOSTS_ENV=${HOSTS_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hosts.env}
[ -s "$HOSTS_ENV" ] || { echo "missing $HOSTS_ENV (copy launcher/hosts.env.example to hosts.env and edit it)" >&2; exit 2; }
while IFS= read -r _hl || [ -n "$_hl" ]; do
  case "$_hl" in ''|'#'*) continue;; esac
  _hk=${_hl%%=*}
  [[ "$_hk" =~ ^[A-Z_][A-Z0-9_]*$ ]] || { echo "$HOSTS_ENV: not a KEY=VALUE line: $_hl" >&2; exit 2; }
  [ -n "${!_hk+x}" ] || eval "export $_hl"
done < "$HOSTS_ENV"
unset _hl _hk
# rank_ssh <rank>: the ssh target for a rank (RANKn_SSH, else [SSH_USER@]RANKn_IP)
rank_ssh() { local s="RANK${1}_SSH" i="RANK${1}_IP"; echo "${!s:-${SSH_USER:+$SSH_USER@}${!i}}"; }
