#!/usr/bin/env bash
# Finish the Zenodo release once every file has been uploaded.
#
#   bash finalize_zenodo.sh <deposition_id> <archive_dir>
#
# It checks that all three expected files are on the server with the expected
# sizes, publishes the deposition, prints the DOIs, and leaves a file with them
# so that fill_doi.py can substitute them into the manuscript.
set -uo pipefail

DEP_ID="${1:?usage: $0 <deposition_id>}"
ARCHIVE_DIR="${2:-./upload}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TOKEN="${ZENODO_TOKEN:-}"
[ -z "$TOKEN" ] && [ -f "$HERE/.ztok" ] && TOKEN="$(cat "$HERE/.ztok")"
[ -z "$TOKEN" ] && { echo "no token" >&2; exit 1; }

API="https://zenodo.org/api"
UA="ProtocolA-ZenodoPublisher/1.0 (academic release)"

# --- expected files, read from the local archive so the check cannot drift ---
python3 - "$ARCHIVE_DIR" << 'PY' > /tmp/expected.json
import hashlib, json, os, sys
d = sys.argv[1]
out = []
for n in sorted(os.listdir(d)):
    p = os.path.join(d, n)
    if os.path.isfile(p):
        h = hashlib.md5()
        with open(p, "rb") as fp:
            for c in iter(lambda: fp.read(1 << 20), b""):
                h.update(c)
        out.append({"name": n, "size": os.path.getsize(p), "md5": h.hexdigest()})
json.dump(out, open("/tmp/expected.json", "w"))
PY

echo "=== expected files ==="
cat /tmp/expected.json | python3 -m json.tool | head -20

# --- what the server has ---
curl -sL -A "$UA" -H "Authorization: Bearer $TOKEN" \
  "$API/deposit/depositions/$DEP_ID" -o /tmp/dep.json
python3 - << 'PY'
import json, sys
exp = json.load(open("/tmp/expected.json"))
dep = json.load(open("/tmp/dep.json"))
have = {f["filename"]: f for f in dep.get("files", [])}
bad = []
for e in exp:
    f = have.get(e["name"])
    if not f:
        bad.append(f'{e["name"]}: MISSING')
    elif f.get("filesize") != e["size"]:
        bad.append(f'{e["name"]}: size {f.get("filesize")} != {e["size"]}')
    elif f.get("checksum") != e["md5"]:
        bad.append(f'{e["name"]}: md5 {f.get("checksum")} != {e["md5"]}')
    else:
        print(f'  OK  {e["name"]}  {e["size"]/1024/1024:.2f} MB  md5 {e["md5"]}')
extra = [n for n in have if n not in {e["name"] for e in exp}]
for n in extra:
    print(f'  extra on the server (will be removed): {n}')
if bad:
    print("\nPROBLEMS:")
    for b in bad:
        print("  " + b)
    sys.exit(1)
print("\nall files present and verified")
PY
rc=$?
[ $rc -ne 0 ] && { echo "not publishing"; exit $rc; }

# --- remove any stray files ---
python3 - << 'PY' > /tmp/extra.txt
import json
exp = {e["name"] for e in json.load(open("/tmp/expected.json"))}
dep = json.load(open("/tmp/dep.json"))
for f in dep.get("files", []):
    if f["filename"] not in exp:
        print(f["filename"])
PY
while read -r n; do
  [ -z "$n" ] && continue
  echo "removing stray file: $n"
  curl -sL -A "$UA" -X DELETE -H "Authorization: Bearer $TOKEN" \
    "$API/deposit/depositions/$DEP_ID/files/$n" -o /dev/null
done < /tmp/extra.txt

# --- publish ---
echo
echo "=== publishing ==="
curl -sL -A "$UA" -H "Authorization: Bearer $TOKEN" \
  -X POST "$API/deposit/depositions/$DEP_ID/actions/publish" -o /tmp/pub.json -w "  HTTP %{http_code}\n"
python3 - << 'PY'
import json
d = json.load(open("/tmp/pub.json"))
if "doi" in d:
    print("  DOI        :", d.get("doi"))
    print("  concept DOI:", d.get("conceptdoi"))
    print("  URL        :", d.get("links", {}).get("html"))
    print("  record id  :", d.get("id"))
    with open("/tmp/zenodo_ids.env", "w") as fp:
        fp.write(f'VERSION_DOI={d.get("doi")}\n')
        fp.write(f'CONCEPT_DOI={d.get("conceptdoi")}\n')
        fp.write(f'RECORD_ID={d.get("id")}\n')
    print("\n  ids written to /tmp/zenodo_ids.env")
else:
    print("  publish failed:", d)
PY
