#!/bin/sh
set -e

chown -R ebay:ebay /data 2>/dev/null || true

exec su -s /bin/sh -c "exec $*" ebay
