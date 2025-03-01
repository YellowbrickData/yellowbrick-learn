#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <source-tarball>"
    exit 1
fi

SOURCE_TARBALL="$1"
TARGET_TARBALL="${SOURCE_TARBALL/ybtools/ybload}"
TEMP_DIR="${SOURCE_TARBALL}-temp"
SKINNY_DIR="${TARGET_TARBALL}-temp"
LIB_DIR="${TEMP_DIR}/ybtools/lib"

BIN_FILES=(
    "bin/setup-yb-env.sh"
    "bin/ybload"
)

LIB_FILES=(
    "aws-java-sdk-core"
    "aws-java-sdk-s3"
    "client-cli"
    "client-common"
    "client-database"
    "client-token"
    "commons-collections"
    "commons-compress"
    "failureaccess"
    "google-http-client"
    "google-http-client-gson"
    "google-oauth-client"
    "google-oauth-client-java6"
    "google-oauth-client-jetty"
    "gson"
    "guava"
    "hadoop-common"
    "httpcore"
    "istack-commons-runtime"
    "jackson-databind"
    "jakarta.activation"
    "jakarta.xml.bind-api"
    "javax.ws.rs-api"
    "jaxb-api"
    "jaxb-runtime"
    "log4j-1.2-api"
    "log4j-api"
    "log4j-core"
    "log4j-slf4j-impl"
    "lz4-java"
    "parquet-column"
    "parquet-common"
    "parquet-encoding"
    "parquet-format-structures"
    "parquet-hadoop"
    "parquet-jackson"
    "postgresql"
    "slf4j-api"
    "slf4j-reload4j"
    "snappy-java"
    "stax-ex"
    "stax2-api"
    "woodstox-core"
    "yb-file-core"
    "yb-file-plugin-s3"
    "yb-log4j"
    "yb-mindeps"
    "yb-optlib"
    "yb-parquet-utils"
    "yb-remote-client"
    "ybload-adapters-base"
    "ybload-adapters-parquet"
    "ybload-api-core"
    "ybload-backend"
    "ybload-cli"
    "ybload-common"
    "ybload-dtypes-parsers"
    "ybload-dtypes-values"
    "ybload-flatfile"
    "ybload-frontend"
)

rm -rf "$TEMP_DIR" "$SKINNY_DIR"
mkdir -p "$SKINNY_DIR/ybload" "$TEMP_DIR"

tar -xzf "$SOURCE_TARBALL" -C "$TEMP_DIR"

RESOLVED_LIB_FILES=()
for base_name in "${LIB_FILES[@]}"; do
    MATCHES=($(ls "$LIB_DIR"/"$base_name"-*.jar 2>/dev/null))

    if [[ ${#MATCHES[@]} -gt 0 ]]; then
        RESOLVED_LIB_FILES+=("${MATCHES[@]}")
    else
        echo "Warning: No matching JAR found for $base_name"
    fi
done

mkdir -p "$SKINNY_DIR/ybload/bin"
for file in "${BIN_FILES[@]}"; do
    cp "$TEMP_DIR/ybtools/$file" "$SKINNY_DIR/ybload/bin/"
done

mkdir -p "$SKINNY_DIR/ybload/lib"
for file in "${RESOLVED_LIB_FILES[@]}"; do
    cp "$file" "$SKINNY_DIR/ybload/lib/"
done

tar -czf "$TARGET_TARBALL" -C "$SKINNY_DIR" .

rm -rf "$TEMP_DIR" "$SKINNY_DIR"

echo "ybload tarball created: $TARGET_TARBALL"
