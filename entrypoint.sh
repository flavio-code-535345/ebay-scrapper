#!/bin/sh
set -e

chown ebay:ebay /data 2>/dev/null || true

exec su -s /bin/sh -c "exec $*" ebay
