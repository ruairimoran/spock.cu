#!/usr/bin/env bash
# Preserve reports outside RUNNER_TEMP before GitHub cleans the build tree.
set -euo pipefail

: "${SPOCK_BUILD_DIR:?Set SPOCK_BUILD_DIR to the CI build directory}"
: "${SPOCK_CI_LOG_DIR:?Set SPOCK_CI_LOG_DIR to a persistent log directory}"
mkdir -p "$SPOCK_CI_LOG_DIR"

# Missing reports are normal when an earlier CI step failed. Preserve the
# relative layout so reports from different tools cannot overwrite each other.
for report in \
    CMakeFiles/CMakeConfigureLog.yaml \
    CMakeFiles/CMakeError.log \
    CMakeFiles/CMakeOutput.log \
    Testing/Temporary \
    examples/networkedControlMpc/test-results \
    sanitizer; do
    if [[ -e "$SPOCK_BUILD_DIR/$report" ]]; then
        if [[ -d "$SPOCK_BUILD_DIR/$report" ]]; then
            mkdir -p "$SPOCK_CI_LOG_DIR/$report"
            cp -a -- "$SPOCK_BUILD_DIR/$report/." "$SPOCK_CI_LOG_DIR/$report/"
        else
            mkdir -p "$(dirname -- "$SPOCK_CI_LOG_DIR/$report")"
            cp -a -- "$SPOCK_BUILD_DIR/$report" "$SPOCK_CI_LOG_DIR/$report"
        fi
    fi
done

{
    printf 'Repository: %s\n' "${GITHUB_REPOSITORY:-local}"
    printf 'Commit: %s\n' "${GITHUB_SHA:-unknown}"
    printf 'Ref: %s\n' "${GITHUB_REF:-unknown}"
    printf 'Run: %s (attempt %s)\n' "${GITHUB_RUN_ID:-local}" "${GITHUB_RUN_ATTEMPT:-1}"
    printf 'Runner: %s\n' "${RUNNER_NAME:-unknown}"
    printf 'Job result: %s\n' "${SPOCK_CI_RESULT:-unknown}"
    printf 'Build directory: %s\n' "$SPOCK_BUILD_DIR"
} > "$SPOCK_CI_LOG_DIR/run-info.txt"

echo "CI logs retained on ${RUNNER_NAME:-this machine}: $SPOCK_CI_LOG_DIR"
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    printf 'CI logs retained on the runner:\n\n```text\n%s\n```\n' "$SPOCK_CI_LOG_DIR" >> "$GITHUB_STEP_SUMMARY"
fi
