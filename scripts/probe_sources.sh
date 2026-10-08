#!/usr/bin/env bash
# Live availability probe of all vacancy sources: HTTP code and response time per URL.
# Not part of check.sh (it goes to the network). Needs only curl.
#
#   bash scripts/probe_sources.sh            # probe from this machine
#   bash scripts/probe_sources.sh --abroad   # also ask check-host.net nodes (us, de, nl)
#
# Exit code is the number of URLs whose HTTP code differs from the expected one.
set -u

UA='ru-it-jobs-radar/0.1 (+https://github.com/alekseevpavel04/ru-it-jobs-radar)'
PAUSE="${PROBE_PAUSE:-2}"
ABROAD=0
[ "${1:-}" = "--abroad" ] && ABROAD=1

CHANNELS="datasciencejobs geekjobs habr_career foranalysts datajobschannel llm_jobs nlp_jobs remoteit jobforjunior job_python"

# source|expected code|url
TARGETS="trudvsem|200|https://opendata.trudvsem.ru/robots.txt
trudvsem|200|https://opendata.trudvsem.ru/api/v1/vacancies?industry=InformationTechnology&limit=1
trudvsem|200|https://opendata.trudvsem.ru/json/industry.json
habr|200|https://career.habr.com/robots.txt
habr|200|https://career.habr.com/api/frontend/vacancies?type=all&sort=date&page=1&per_page=1
habr|200|https://career.habr.com/vacancies?type=all
telegram|404|https://t.me/robots.txt"
for ch in $CHANNELS; do
  TARGETS="$TARGETS
telegram|200|https://t.me/s/$ch"
done
# Without an app key the vacancies method answers 403; that is the expected state until USER_ACTIONS gives a key.
TARGETS="$TARGETS
superjob|200|https://api.superjob.ru/robots.txt
superjob|403|https://api.superjob.ru/2.0/vacancies/?count=1"

failed=0
printf '%-9s %-4s %-4s %8s  %s\n' source exp code time_s url
while IFS='|' read -r src exp url; do
  out=$(curl -sS -o /dev/null -A "$UA" --max-time 90 -w '%{http_code} %{time_total}' "$url" 2>&1)
  code=${out%% *}
  t=${out#* }
  [ "$code" = "$exp" ] || failed=$((failed + 1))
  printf '%-9s %-4s %-4s %8s  %s\n' "$src" "$exp" "$code" "$t" "$url"
  sleep "$PAUSE"
done <<EOF
$TARGETS
EOF

if [ "$ABROAD" = 1 ]; then
  echo
  echo "check-host.net check-http (nodes us1, us2, de1, de4, nl1); result format per node: [ok, time_s, message, code, ip]"
  nodes="node=us1.node.check-host.net&node=us2.node.check-host.net&node=de1.node.check-host.net&node=de4.node.check-host.net&node=nl1.node.check-host.net"
  for url in "https://opendata.trudvsem.ru/api/v1/vacancies?limit=1" \
             "https://career.habr.com/api/frontend/vacancies?type=all&sort=date&page=1&per_page=1" \
             "https://t.me/s/datasciencejobs"; do
    enc=$(printf '%s' "$url" | sed -e 's/%/%25/g' -e 's/&/%26/g' -e 's/?/%3F/g' -e 's/=/%3D/g')
    start=$(curl -sS -A "$UA" -H 'Accept: application/json' "https://check-host.net/check-http?host=${enc}&${nodes}")
    rid=$(printf '%s' "$start" | grep -o '"request_id":"[^"]*"' | cut -d'"' -f4)
    if [ -z "$rid" ]; then
      echo "$url: check-host did not return request_id: $start"
      failed=$((failed + 1))
      continue
    fi
    # trudvsem answers in 10-15 s, the nodes need time before the result is complete
    sleep 25
    echo "$url request_id=$rid https://check-host.net/check-report/$rid"
    curl -sS -A "$UA" -H 'Accept: application/json' "https://check-host.net/check-result/$rid"
    echo
  done
fi

exit "$failed"
