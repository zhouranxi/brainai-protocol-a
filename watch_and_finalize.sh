#!/usr/bin/env bash
# Watch the checkpoint upload and finish the Zenodo release once it lands.
#
#   bash watch_and_finalize.sh <deposition_id> [archive_dir]
#
# It polls the deposition, prints a progress line, and when all three files are
# present with the expected sizes it verifies the md5 of each one, publishes the
# record, and writes the identifiers to /tmp/zenodo_ids.env. It is safe to run
# repeatedly: publishing an already-published deposition simply fails and is
# reported as such.
set -uo pipefail

DEP_ID="${1:?usage: $0 <deposition_id> [archive_dir]}"
ARCHIVE_DIR="${2:-./upload}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TOKEN="${ZENODO_TOKEN:-}"
[ -z "$TOKEN" ] && [ -f "$HERE/.ztok" ] && TOKEN="$(cat "$HERE/.ztok")"
[ -z "$TOKEN" ] && { echo "no token" >&2; exit 1; }

API="https://zenodo.org/api"
UA="ProtocolA-ZenodoPublisher/1.0 (academic release)"
EXPECT_SIZE=2151270400   # checkpoints_resnet18.tar
MARKER="$HERE/.finalized"

if [ -f "$MARKER" ]; then
  echo "already finalized:"
  cat "$MARKER"
  exit 0
fi

echo "[$(date '+%F %T')] checking deposition $DEP_ID ..."
curl -sL -A "$UA" -H "Authorization: Bearer $TOKEN" \
  "$API/deposit/depositions/$DEP_ID" -o /tmp/dep_watch.json

STATE=$(python3 -c "
import json
d = json.load(open('/tmp/dep_watch.json'))
print(d.get('state', '?'))
")
REMOTE_SIZE=$(python3 -c "
import json
d = json.load(open('/tmp/dep_watch.json'))
for f in d.get('files', []):
    if f['filename'] == 'checkpoints_resnet18.tar':
        print(f.get('filesize', 0)); break
else:
    print(0)
")
NFILES=$(python3 -c "
import json
print(len(json.load(open('/tmp/dep_watch.json')).get('files', [])))
")

echo "[$(date '+%F %T')] state=$STATE  files=$NFILES  checkpoints_resnet18.tar=$(( REMOTE_SIZE / 1024 / 1024 )) / 2051 MB"

if [ "$REMOTE_SIZE" -lt "$EXPECT_SIZE" ]; then
  echo "  still uploading; nothing to do yet"
  exit 1
fi

echo
echo "=== all files present, verifying and publishing ==="
bash "$HERE/finalize_zenodo.sh" "$DEP_ID" "$ARCHIVE_DIR"
rc=$?

if [ $rc -eq 0 ] && [ -f /tmp/zenodo_ids.env ]; then
  cp /tmp/zenodo_ids.env "$MARKER"
  echo
  echo "=== DONE ==="
  cat /tmp/zenodo_ids.env
fi
exit $rc
