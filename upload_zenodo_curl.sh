#!/usr/bin/env bash
# Upload the release files to an existing Zenodo deposition using curl.
#
# Why curl and not the Python script: the link to Zenodo from this machine is
# lossy (33% packet loss, 175 ms RTT). Python's `requests` blocks forever when
# the peer half-closes a keep-alive connection mid-upload. curl's
# --speed-limit/--speed-time abort a stalled transfer so the retry loop can
# pick it up, and its --retry handles the rest.
#
# Usage:
#   ZENODO_TOKEN=... ./upload_zenodo_curl.sh <deposition_id> <files_dir>
set -uo pipefail

DEP_ID="${1:?usage: $0 <deposition_id> <files_dir>}"
FILES_DIR="${2:?usage: $0 <deposition_id> <files_dir>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TOKEN="${ZENODO_TOKEN:-}"
if [ -z "$TOKEN" ] && [ -f "$HERE/.ztok" ]; then
  TOKEN="$(cat "$HERE/.ztok")"
fi
if [ -z "$TOKEN" ]; then
  echo "no token: set ZENODO_TOKEN or provide $HERE/.ztok" >&2
  exit 1
fi

API="https://zenodo.org/api"
UA="ProtocolA-ZenodoPublisher/1.0 (academic release)"

# Resolve the bucket URL for this deposition.
BUCKET=$(curl -sL -A "$UA" -H "Authorization: Bearer $TOKEN" \
  "$API/deposit/depositions/$DEP_ID" |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["links"]["bucket"])')
if [ -z "$BUCKET" ]; then
  echo "could not resolve the bucket for deposition $DEP_ID" >&2
  exit 1
fi
echo "bucket: $BUCKET"

# Files already present on the server are skipped.
existing=$(curl -sL -A "$UA" -H "Authorization: Bearer $TOKEN" \
  "$API/deposit/depositions/$DEP_ID" |
  python3 -c 'import json,sys; print(" ".join(f["filename"] for f in json.load(sys.stdin).get("files",[])))')
echo "already on the server: $existing"

for f in "$FILES_DIR"/*; do
  [ -f "$f" ] || continue
  name=$(basename "$f")
  if echo " $existing " | grep -q " $name "; then
    echo "[skip] $name (already uploaded)"
    continue
  fi

  size=$(stat -c%s "$f")
  echo "[put ] $name ($(( size / 1024 / 1024 )) MB)"

  attempt=1
  max=40
  while [ "$attempt" -le "$max" ]; do
    # --speed-limit/--speed-time: abort if throughput drops below 10 KB/s for
    # 60 s, which is what a half-closed connection looks like from here.
    code=$(curl -sL -A "$UA" \
      -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/octet-stream" \
      --speed-limit 10240 --speed-time 60 \
      --connect-timeout 30 --max-time 7200 \
      --retry 0 \
      --upload-file "$f" \
      -o /tmp/put_out.txt -w '%{http_code}' \
      "$BUCKET/$name")
    rc=$?
    if [ "$rc" = "0" ] && [ "$code" = "200" ] || [ "$code" = "201" ]; then
      echo "  ok (attempt $attempt, HTTP $code)"
      break
    fi
    attempt=$((attempt + 1))
    echo "  attempt $attempt failed (curl rc=$rc, HTTP $code); retrying in 5 s"
    if [ "$attempt" -le "$max" ]; then sleep 5; fi
  done

  if [ "$attempt" -gt "$max" ]; then
    echo "  !! giving up on $name after $max attempts" >&2
    exit 1
  fi
done

echo
echo "Files now on the server:"
curl -sL -A "$UA" -H "Authorization: Bearer $TOKEN" \
  "$API/deposit/depositions/$DEP_ID" |
  python3 -c 'import json,sys
for f in json.load(sys.stdin).get("files",[]):
    print(f"  {f[\"filename\"]}  {f.get(\"filesize\",0)/1024/1024:.2f} MB  md5={f.get(\"checksum\",\"\")}")'
