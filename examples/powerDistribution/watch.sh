#!/bin/bash

# --- Configuration ---
TIME_SCRIPT="./time.sh" # Assuming time.sh is in the same dir as launcher
WATCHDOG_SCRIPT="../watchdog/memory.sh" # Relative to launcher's location

MAIN_PID_FILE="/tmp/asdf_main_pid.$$" # Store MAIN_PID in a file

# --- Cleanup Function ---
# This function will be called on EXIT or when SIGINT/SIGTERM is received by the launcher
# shellcheck disable=SC2317
cleanup() {
    echo "Launcher: Cleanup called..."
    if [ -f "$MAIN_PID_FILE" ]; then
        STORED_MAIN_PID=$(cat "$MAIN_PID_FILE")
        if [ -n "$STORED_MAIN_PID" ] && ps -p "$STORED_MAIN_PID" > /dev/null; then
            echo "Launcher: Main job (PID $STORED_MAIN_PID from file) seems to be running."
            echo "Launcher: Attempting to terminate process group of PID $STORED_MAIN_PID..."
            # Send SIGTERM to the entire process group of MAIN_PID
            # The '-' before STORED_MAIN_PID is crucial for signaling the group
            if kill -TERM -- "-$STORED_MAIN_PID"; then
                echo "Launcher: Sent SIGTERM to process group -$STORED_MAIN_PID."
                sleep 2 # Grace period
                if ps -p "$STORED_MAIN_PID" > /dev/null; then # Check main PID specifically
                    echo "Launcher: Main PID $STORED_MAIN_PID still alive after SIGTERM to group. Sending SIGKILL to group."
                    kill -KILL -- "-$STORED_MAIN_PID"
                    sleep 1
                     if ps -p "$STORED_MAIN_PID" > /dev/null; then
                        echo "Launcher: WARNING - Main PID $STORED_MAIN_PID still alive after SIGKILL to group." >&2
                    else
                        echo "Launcher: Main PID $STORED_MAIN_PID terminated after SIGKILL to group."
                    fi
                else
                    echo "Launcher: Main PID $STORED_MAIN_PID terminated after SIGTERM to group."
                fi
            else
                echo "Launcher: Failed to send SIGTERM to process group -$STORED_MAIN_PID, or process was already gone."
            fi
        elif [ -n "$STORED_MAIN_PID" ]; then
            echo "Launcher: Main job (PID $STORED_MAIN_PID from file) was not found."
        fi
        rm -f "$MAIN_PID_FILE"
    else
        echo "Launcher: No MAIN_PID file found for cleanup."
    fi

    # Also ensure watchdog is killed if it's still running
    # This is a bit trickier as we don't have its PID directly stored here
    # in a way that's easily accessible in the trap from a different context.
    # A more robust solution would involve the watchdog also writing a PID file.
    # For now, we rely on the watchdog's own trap for SIGINT/SIGTERM.
    echo "Launcher: Cleanup finished."
}

# Trap EXIT, SIGINT, SIGTERM signals to run the cleanup function
trap cleanup EXIT SIGINT SIGTERM

# --- Sanity checks for scripts ---
if [ ! -f "$TIME_SCRIPT" ]; then
    echo "Error: Main script '$TIME_SCRIPT' not found." >&2
    exit 1
fi
if [ ! -x "$TIME_SCRIPT" ]; then
    echo "Error: Main script '$TIME_SCRIPT' is not executable. Please use 'chmod +x $TIME_SCRIPT'." >&2
    exit 1
fi
if [ ! -f "$WATCHDOG_SCRIPT" ]; then
    echo "Error: Watchdog script '$WATCHDOG_SCRIPT' not found." >&2
    exit 1
fi
if [ ! -x "$WATCHDOG_SCRIPT" ]; then
    echo "Error: Watchdog script '$WATCHDOG_SCRIPT' is not executable. Please use 'chmod +x $WATCHDOG_SCRIPT'." >&2
    exit 1
fi

echo "Launcher: Starting main job '$TIME_SCRIPT' in the background..."
# Start time.sh in a way that it becomes a process group leader (optional but can help)
# However, simply backgrounding it often puts it in its own group or shares the launcher's.
# The key is that 'kill -TERM -- -PID' targets the group of PID.
( "$TIME_SCRIPT" ) & # Subshell for `time.sh`
MAIN_PID=$!
echo "$MAIN_PID" > "$MAIN_PID_FILE" # Store MAIN_PID

# Check if MAIN_PID was captured and process started
if [ -z "$MAIN_PID" ] || ! ps -p "$MAIN_PID" > /dev/null; then
    echo "Error: Failed to start '$TIME_SCRIPT' or get its PID. Current PID: '$MAIN_PID'." >&2
    rm -f "$MAIN_PID_FILE"
    exit 1 # This will trigger the EXIT trap
fi

echo "Launcher: Main job '$TIME_SCRIPT' started with PID: $MAIN_PID (Process Group ID should be the same)"

# Disown the job (MAIN_PID). This is primarily so it doesn't get SIGHUP if the launcher exits
# *without* the trap running (e.g., if the launcher itself is SIGKILLed).
# The trap should handle normal termination.
disown "$MAIN_PID"
echo "Launcher: Main job PID $MAIN_PID has been disowned."

# Export MAIN_PID so the child script (watchdog.sh) can access it
export MAIN_PID

echo "Launcher: Starting watchdog '$WATCHDOG_SCRIPT' for PID $MAIN_PID..."
"$WATCHDOG_SCRIPT" # Run watchdog in the foreground of the launcher
watchdog_exit_status=$?

echo "Launcher: Watchdog script exited with status $watchdog_exit_status."

# The EXIT trap will handle cleanup based on MAIN_PID.
# If watchdog exited with 0, it means it handled MAIN_PID. The trap will see MAIN_PID is gone.
# If watchdog exited non-zero, the trap will try to kill MAIN_PID (and its group).

echo "Launcher: Exiting via normal script completion (trap will run)."
# Exit status of launcher will be determined by the last command or explicit exit in trap
# For now, let's stick to the watchdog's status or what the trap decides.
exit "$watchdog_exit_status"