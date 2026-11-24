#!/bin/bash
# usage: queue.sh FILE  -- each line: NAME key=value ...; runs 2 in parallel, 1 thread each
cd "$(dirname "$0")"
mkdir -p logs
grep -v '^#' "$1" | grep -v '^\s*$' | xargs -P 2 -L 1 bash -c 'python3 exp.py "$0" "$@" num_threads=1 > logs/$0.log 2>&1; echo done $0'
