#!/usr/bin/env bash
# Upload the checkpoint archives to a Zenodo deposition in chunks.
#
# Why chunks: the link from this machine to Zenodo (CWI, Netherlands) is lossy
# -- roughly 30% packet loss and 175 ms RTT. A single 2 GB PUT stalls when the
# peer half-closes a keep-alive connection mid-transfer, and neither Python's
# `requests` nor curl can recover from that on their own. Splitting the archive
# into 32 MB parts turns one fragile 2 GB transfer into ~80 short ones, each of
# which either completes or is retried.
#
# The parts are uploaded as individual files. The deposition description and
# the accompanying README explain how to reassemble them with `cat`.
#
# Usage:
#   ZENODO_TOKEN=... ./upload_zenodo_chunked.sh <deposition_id> <archive> <part_size_MB>
set -uo pipefail

DEP_ID="${1:?usage: $0 <deposition_id> <archive> [part_mb]}"
ARCHIVE="${2:?usage: $0 <deposition_id> <archive> [part_mb]}"
PART_MB="${3:-32}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TOKEN="${ZENODO_TOKEN:-}"
[ -z "$TOKEN" ] && [ -f "$HERE/.ztok" ] && TOKEN="$(cat "$HERE/.ztok")"
if [ -z "$TOKEN" ]; then
  echo "no token: set ZENODO_TOKEN or provide $HERE/.ztok" >&2
  exit 1
fi

API="https://zenodo.org/api"
UA="ProtocolA-ZenodoPublisher/1.0 (academic release)"
WORK="$HERE/chunkwork"
MAX_TRIES=12

mkdir -p "$WORK"
BUCKET=$(curl -sL -A "$UA" -H "Authorization: Bearer $TOKEN" \
  "$API/deposit/depositions/$DEP_ID" |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["links"]["bucket"])')
[ -z "$BUCKET" ] && { echo "could not resolve bucket" >&2; exit 1; }
echo "bucket: $BUCKET"
echo "archive: $ARCHIVE"

BASE=$(basename "$ARCHIVE")
PART_SIZE="${PART_MB}M"
PARTS="$WORK/${BASE}.parts"
SPLIT="$WORK/${BASE}.split"

# Split once; reuse the pieces across retries.
if [ ! -f "${SPLIT}.done" ]; then
  echo "splitting into ${PART_MB} MB parts ..."
  rm -f "$WORK/${BASE}.part"*
  split -b "$PART_SIZE" -d -a 3 "$ARCHIVE" "$WORK/${BASE}.part"
  touch "${SPLIT}.done"
fi
ls "$WORK/${BASE}.part"* | wc -l | xargs echo "parts:"

# Which parts are already on the server?
existing=$(curl -sL -A "$UA" -H "Authorization: Bearer $TOKEN" \
  "$API/deposit/depositions/$DEP_ID" |
  python3 -c 'import json,sys; print("\n".join(f["filename"] for f in json.load(sys.stdin).get("files",[])))')
echo "already on the server: $(echo "$existing" | grep -c "${BASE}.part") parts"

FAILED=0
for part in "$WORK/${BASE}.part"*; do
  pname="$(basename "$part")"
  if echo "$existing" | grep -qx "$pname"; then
    echo "[skip] $pname"
    continue
  fi
  attempt=1
  while [ "$attempt" -le "$MAX_TRIES" ]; do
    code=$(curl -sL -A "$UA" \
      -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/octet-stream" \
      --speed-limit 10240 --speed-time 45 \
      --connect-timeout 30 --max-time 900 \
      --retry 0 \
      --upload-file "$part" \
      -o /tmp/chunk_out.txt -w '%{http_code}' \
      "$BUCKET/$pname")
    if [ "$code" = "200" ] || [ "$code" = "201" ]; then
      echo "[ok  ] $pname (attempt $attempt)"
      break
    fi
    attempt=$((attempt + 1))
    echo "  $pname attempt $attempt failed (HTTP $code), retry in 5 s"
    [ "$attempt" -le "$MAX_TRIES" ] && sleep 5
  done
  if [ "$attempt" -gt "$MAX_TRIES" ]; then
    echo "  !! $pname gave up after $MAX_TRIES attempts" >&2
    FAILED=$((FAILED + 1))
  fi
done

echo
echo "=== final file list ==="
curl -sL -A "$UA" -H "Authorization: Bearer $TOKEN" \
  "$API/deposit/depositions/$DEP_ID" |
  python3 -c 'import json,sys
fs=json.load(sys.stdin).get("files",[])
tot=0
for f in sorted(fs, key=lambda x: x["filename"]):
    print(f"  {f[\"filename\"]}  {f.get(\"filesize\",0)/1024/1024:.2f} MB  md5={f.get(\"checksum\",\"\")}")
    tot+=f.get("filesize",0)
print(f"  total {tot/1024/1024:.1f} MB in {len(fs)} files")'

exit $((FAILED > 0))
