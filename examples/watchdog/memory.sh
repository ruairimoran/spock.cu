#!/bin/bash

# Configuration
readonly SLEEP_INTERVAL=30      # Check every 30 seconds
readonly LOW_MEM_THRESHOLD=2   # Kill if available memory is <= 2%
# MAIN_PID should be set in the environment.

cleanup_and_exit_watchdog() {
    local exit_code=$1
    local message=$2
    echo "---"
    echo "Watchdog: $message"
    echo "Memory monitor is stopping."
    exit "$exit_code"
}

# --- Sanity Checks ---
if [[ -z "$MAIN_PID" ]]; then
    cleanup_and_exit_watchdog 1 "Error: MAIN_PID environment variable is not set."
fi

if ! ps -p "$MAIN_PID" > /dev/null; then
    cleanup_and_exit_watchdog 1 "Error: Process with PID $MAIN_PID not found at startup."
fi

echo "Starting memory monitor for PID $MAIN_PID."
echo "Will check every $SLEEP_INTERVAL seconds."
echo "If available memory drops to $LOW_MEM_THRESHOLD% or less, PID $MAIN_PID will be signaled."
echo "---"

# Trap signals to allow for a custom exit if watchdog is interrupted
trap 'cleanup_and_exit_watchdog 2 "Watchdog interrupted by signal."' INT TERM QUIT

while sleep "$SLEEP_INTERVAL"; do
    if ! ps -p "$MAIN_PID" > /dev/null; then
        cleanup_and_exit_watchdog 0 "Monitored process PID $MAIN_PID no longer exists."
    fi

    available_mem_percent=$(LC_ALL=C free | awk '
        /^Mem:/ {
            total = $2; available = $7
            if (total > 0) { printf "%.0f", (available / total) * 100 } else { print "0" }
            exit
        }')

    if ! [[ "$available_mem_percent" =~ ^[0-9]+$ ]]; then
        echo "Warning: Could not reliably determine available memory. Received: '$available_mem_percent'. Skipping this check." >&2
        continue
    fi

    echo "Current available memory: ${available_mem_percent}%"

    if [[ "$available_mem_percent" -le "$LOW_MEM_THRESHOLD" ]]; then
        echo "Low memory detected: ${available_mem_percent}% <= ${LOW_MEM_THRESHOLD}%."
        echo "Attempting to terminate PID $MAIN_PID."

        # Check if process still exists right before trying to kill
        if ! ps -p "$MAIN_PID" > /dev/null; then
            cleanup_and_exit_watchdog 0 "Process PID $MAIN_PID no longer exists when attempting to kill. No action taken."
        fi

        echo "Sending SIGTERM to PID $MAIN_PID."
        if kill -TERM "$MAIN_PID"; then
            echo "SIGTERM sent to PID $MAIN_PID. Waiting for graceful shutdown..."
            # Wait for a few seconds for graceful termination
            grace_period=3 # seconds
            for (( i=0; i<grace_period; i++ )); do
                if ! ps -p "$MAIN_PID" > /dev/null; then
                    cleanup_and_exit_watchdog 0 "PID $MAIN_PID terminated gracefully after SIGTERM."
                fi
                sleep 1
            done

            # If still alive after grace period, send SIGKILL
            echo "PID $MAIN_PID still alive after ${grace_period}s. Sending SIGKILL."
            if ps -p "$MAIN_PID" > /dev/null; then # Check again before SIGKILL
                if kill -KILL "$MAIN_PID"; then
                    echo "SIGKILL sent to PID $MAIN_PID."
                    sleep 1 # Give SIGKILL a moment to act
                    if ! ps -p "$MAIN_PID" > /dev/null; then
                        cleanup_and_exit_watchdog 0 "PID $MAIN_PID terminated after SIGKILL."
                    else
                        cleanup_and_exit_watchdog 1 "ERROR: PID $MAIN_PID still alive after SIGKILL. Manual intervention may be required."
                    fi
                else
                    # This can happen if process died just between ps check and kill -KILL
                    if ! ps -p "$MAIN_PID" > /dev/null; then
                        cleanup_and_exit_watchdog 0 "Failed to send SIGKILL to PID $MAIN_PID, but it was already gone."
                    else
                        cleanup_and_exit_watchdog 1 "ERROR: Failed to send SIGKILL to PID $MAIN_PID (and it's still alive)."
                    fi
                fi
            else
                 cleanup_and_exit_watchdog 0 "PID $MAIN_PID was gone before SIGKILL could be sent."
            fi
        else
            # kill -TERM failed
            if ! ps -p "$MAIN_PID" > /dev/null; then
                 cleanup_and_exit_watchdog 0 "Failed to send SIGTERM to PID $MAIN_PID, but it was already gone."
            else
                 cleanup_and_exit_watchdog 1 "ERROR: Failed to send SIGTERM to PID $MAIN_PID (and it's still alive). Permissions?"
            fi
        fi
        # Should have exited via cleanup_and_exit_watchdog by now
    fi
done

cleanup_and_exit_watchdog 3 "Watchdog loop exited unexpectedly."
