#!/usr/bin/env bash
# Repeats the tests a failed ctest run failed as GitHub annotations.
#
#   tools/ci_failed_tests.sh <build directory>
#
# A job's log can be read only by someone signed in to GitHub, while its
# annotations are public: the check-run API returns them to anyone. A Windows
# test failure that nobody without the log could name sat on dev for two pushes.
# So after a failed test step this prints, per failed test, its name and the
# first lines of its failure from Testing/Temporary/LastTest.log, as ::error
# workflow commands.
set -u
dir=${1:?usage: ci_failed_tests.sh <build directory>}
failed="$dir/Testing/Temporary/LastTestsFailed.log"
log="$dir/Testing/Temporary/LastTest.log"

# workflow-command message escaping: %, then CR and LF
escape() {
    local s=${1//%/%25}
    s=${s//$'\r'/%0D}
    printf '%s' "${s//$'\n'/%0A}"
}

if [ ! -s "$failed" ]; then
    echo "::error title=ctest::the test step failed, but $failed names no test (did the build or the test discovery fail?)"
    exit 0
fi

count=0
while IFS= read -r line; do
    line=${line%$'\r'}   # ctest writes its logs in text mode: CRLF on Windows
    [ -z "$line" ] && continue
    name=${line#*:}
    count=$((count + 1))
    # a job shows at most 10 error annotations per step; keep one for the summary
    if [ "$count" -gt 9 ]; then
        echo "::error title=ctest::$(escape "$(($(grep -c . "$failed") - 9)) more failed tests are only in the log")"
        break
    fi
    detail=""
    if [ -f "$log" ]; then
        # the test's own output, from its "N/M Test: <name>" header to <end of output>,
        # from the first failed assertion on
        detail=$(awk -v name="$name" '
            { sub(/\r$/, "") }
            !on && /^[0-9]+\/[0-9]+ Test: / && substr($0, index($0, "Test: ") + 6) == name { on = 1; next }
            on && !out && /^Output:$/ { out = 1; next }
            out && /^<end of output>$/ { exit }
            out && (seen || /FAILED|unexpected exception|fatal error condition/) { seen = 1; print }
        ' "$log" | head -n 15)
    fi
    message=$name
    [ -n "$detail" ] && message+=$'\n'$detail
    echo "::error title=Failed test::$(escape "$message")"
done < "$failed"
