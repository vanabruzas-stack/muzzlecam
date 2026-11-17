#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
output="${1:-$root/dist}"
bundle_id="${BUNDLE_ID:-com.muzzlezoom.camera}"
if [[ ! "$bundle_id" =~ ^[a-zA-Z0-9-]+(\.[a-zA-Z0-9-]+)+$ ]]; then
  printf 'Invalid BUNDLE_ID: %s\n' "$bundle_id" >&2
  exit 2
fi

if [[ "$(uname -s)" != Darwin ]] || ! command -v xcodebuild >/dev/null; then
  printf 'Building an iOS IPA requires macOS with Xcode 27.\n' >&2
  exit 2
fi
xcodebuild -version
sdk="$(xcrun --sdk iphoneos --show-sdk-version)"
if [[ "$sdk" != 27.* ]]; then
  printf 'Select Xcode with an iOS 27 SDK. Selected SDK: %s\n' "$sdk" >&2
  exit 2
fi
if ! xcrun --sdk iphoneos metal --version; then
  xcodebuild -downloadComponent MetalToolchain
  xcrun --kill-cache
  xcrun --sdk iphoneos metal --version
fi

mkdir -p "$output"
output="$(cd "$output" && pwd)"
build="$(mktemp -d "${TMPDIR:-/tmp}/muzzlezoom.XXXXXXXX")"
trap 'rm -rf "$build"' EXIT
(cd "$root/App/Models" && shasum -a 256 -c SHA256SUMS.txt)

xcrun swiftc -O -parse-as-library "$root/App/Imaging/Registration.swift" \
  "$root/App/Camera/CameraTypes.swift" "$root/Tests/RegistrationCheck.swift" \
  -o "$build/check-registration"
"$build/check-registration"

xcodebuild build \
  -project "$root/MuzzleZoom.xcodeproj" -scheme MuzzleZoom \
  -configuration Release -sdk iphoneos -destination 'generic/platform=iOS' \
  -derivedDataPath "$build/DerivedData" \
  ARCHS=arm64 CODE_SIGNING_ALLOWED=NO CODE_SIGNING_REQUIRED=NO CODE_SIGN_IDENTITY="" \
  PRODUCT_BUNDLE_IDENTIFIER="$bundle_id" COMPILER_INDEX_STORE_ENABLE=NO

product="$build/DerivedData/Build/Products/Release-iphoneos/MuzzleZoom.app"
binary="$product/MuzzleZoom"
test -x "$binary"
[[ "$(xcrun lipo -archs "$binary")" == "arm64" ]]
xcrun vtool -show-build "$binary" | grep -Eq 'platform[[:space:]]+IOS([[:space:]]|$)'
[[ "$(plutil -extract CFBundleIdentifier raw -o - "$product/Info.plist")" == "$bundle_id" ]]
test -d "$product/MuzzlePreviewSR.mlmodelc"
test -d "$product/MuzzlePhotoSR.mlmodelc"
test -d "$product/MuzzleExtremeSR.mlmodelc"
test -f "$product/default.metallib"
test ! -e "$product/embedded.mobileprovision"
test ! -d "$product/_CodeSignature"

# Exercise the compiled models through Core ML on the Apple runner.
# iPhone Neural Engine performance still requires the physical phone.
mkdir "$build/HostModels"
for model in MuzzlePreviewSR MuzzlePhotoSR MuzzleExtremeSR; do
  xcrun coremlcompiler compile "$root/App/Models/$model.mlmodel" "$build/HostModels"
done
xcrun swiftc -O -parse-as-library "$root/App/Camera/CameraTypes.swift" \
  "$root/App/Imaging/Registration.swift" "$root/App/Imaging/ImageTools.swift" \
  "$root/Tests/CoreMLCheck.swift" -o "$build/check-models"
"$build/check-models" "$build/HostModels"

mkdir "$build/Payload"
ditto "$product" "$build/Payload/MuzzleZoom.app"
ipa="$output/MuzzleZoom-unsigned.ipa"
ditto -c -k --keepParent "$build/Payload" "$build/MuzzleZoom-unsigned.ipa"
unzip -tq "$build/MuzzleZoom-unsigned.ipa"
unzip -Z1 "$build/MuzzleZoom-unsigned.ipa" > "$build/entries.txt"
grep -qx 'Payload/MuzzleZoom.app/MuzzleZoom' "$build/entries.txt"
mv -f "$build/MuzzleZoom-unsigned.ipa" "$ipa"
(cd "$output" && shasum -a 256 MuzzleZoom-unsigned.ipa > SHA256SUMS.txt)
printf 'Unsigned IPA: %s\n' "$ipa"
