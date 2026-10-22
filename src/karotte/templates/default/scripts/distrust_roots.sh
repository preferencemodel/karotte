#!/usr/bin/env bash
# Removes trust in the emSign roots and AAA Certificate Services, whose subordinate
# CAs had private keys compromised (Mozilla, 2026-10-20). Covers the OS trust store
# and every Mozilla-derived bundle a Python package ships (certifi, pip's vendored
# certifi, httplib2), since those ignore the OS store.
#
# Run as root after every package install that could bring a bundle back: an apt or
# dnf install regenerates the OS store but keeps the exclusion, while a `uv sync` that
# reinstalls certifi does not, so it has to run again after it.
set -euo pipefail

LABEL_PATTERN='emSign|AAA Certificate Services'

if [ -f /etc/ca-certificates.conf ]; then
    # Debian/Ubuntu: a leading "!" deselects a cert and survives later upgrades. Matched
    # on subject, not file name (Debian ships AAA as Comodo_AAA_Services_root.crt).
    # openssl is there because ca-certificates depends on it.
    grep -E '^mozilla/' /etc/ca-certificates.conf | while read -r cert; do
        if openssl x509 -noout -subject -in "/usr/share/ca-certificates/$cert" | grep -qE "$LABEL_PATTERN"; then
            sed -i "s#^$cert\$#!$cert#" /etc/ca-certificates.conf
        fi
    done
    update-ca-certificates --fresh >/dev/null
elif command -v trust >/dev/null && [ -d /etc/pki/ca-trust/source/blocklist ]; then
    # Fedora/RHEL: p11-kit's blocklist overrides the anchors shipped in /usr/share.
    trust list --filter=ca-anchors |
        awk -v pat="$LABEL_PATTERN" '/^pkcs11:/ { uri = $0 } $1 == "label:" && $0 ~ pat { print uri }' |
        while read -r uri; do
            out="/etc/pki/ca-trust/source/blocklist/$(echo "$uri" | md5sum | cut -c1-16).pem"
            trust extract --overwrite --format=pem-bundle --filter="$uri" "$out"
        done
    update-ca-trust extract
fi

# Bundles in certifi's format: one blank-line-separated block per cert, labelled.
bundles() {
    if command -v find >/dev/null; then
        # find exits non-zero on any unreadable path; that is not a failure here.
        # Container image stores are skipped: editing their unpacked layers in place
        # changes images behind the runtime's back.
        find / -xdev \( -path /var/lib/containerd -o -path /var/lib/buildkit -o -path /var/lib/docker \
            -o -path /var/lib/containers \) -prune -o \
            \( -path '*/certifi/cacert.pem' -o -path '*/httplib2/cacerts.txt' \) -type f -print 2>/dev/null || true
    else
        # Minimal images (amazonlinux) ship without findutils.
        local top f
        shopt -s globstar nullglob
        for top in /*/; do
            case "$top" in /proc/ | /sys/ | /dev/) continue ;; esac
            for f in "$top"**/certifi/cacert.pem "$top"**/httplib2/cacerts.txt; do
                if [ -f "$f" ] && [ ! -L "$f" ]; then echo "$f"; fi
            done
        done
    fi
}

bundles | while read -r bundle; do
    awk -v RS= -v ORS='\n\n' -v pat="# Label: \"($LABEL_PATTERN)" '$0 !~ pat' "$bundle" >"$bundle.tmp"
    cat "$bundle.tmp" >"$bundle"
    rm "$bundle.tmp"
done

# Fail the build rather than ship an image that still trusts them.
remaining=$(
    {
        if command -v trust >/dev/null; then
            trust list --filter=ca-anchors | grep '^ *label:'
        elif [ -f /etc/ssl/certs/ca-certificates.crt ]; then
            openssl crl2pkcs7 -nocrl -certfile /etc/ssl/certs/ca-certificates.crt | openssl pkcs7 -print_certs -noout
        fi
        bundles | while read -r bundle; do grep -H '^# Label' "$bundle"; done
    } | grep -E "$LABEL_PATTERN" || true
)
if [ -n "$remaining" ]; then
    echo "distrust_roots.sh: still trusted:" >&2
    echo "$remaining" >&2
    exit 1
fi
