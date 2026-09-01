from faster_whisper import WhisperModel
import sys
import os
import subprocess
import time
import threading


# ============================================================
# Configuration
# ============================================================

MODEL_SIZE = "small"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"


# ============================================================
# Check arguments
# ============================================================

if len(sys.argv) < 2:

    print("Usage:")
    print('  python3 transcribe.py "video.mp4"')
    sys.exit(1)


video = sys.argv[1]


# ============================================================
# Check video
# ============================================================

if not os.path.isfile(video):

    print()
    print("ERROR: File not found:")
    print(video)
    print()

    sys.exit(1)


# ============================================================
# Get video duration
# ============================================================

def get_duration(filename):

    try:

        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                filename
            ],
            capture_output=True,
            text=True
        )

        return float(
            result.stdout.strip()
        )

    except Exception as e:

        print()
        print(
            "ERROR: Could not determine video duration."
        )
        print(e)
        print()

        sys.exit(1)


duration = get_duration(video)


# ============================================================
# Format time
# ============================================================

def format_time(seconds):

    seconds = max(
        0,
        int(seconds)
    )

    hours = seconds // 3600

    minutes = (
        seconds % 3600
    ) // 60

    secs = seconds % 60

    return (
        f"{hours:02}:"
        f"{minutes:02}:"
        f"{secs:02}"
    )


# ============================================================
# Animated indicator
# ============================================================

spinner_running = False


def spinner(message):

    global spinner_running

    symbols = [
        "⠋",
        "⠙",
        "⠹",
        "⠸",
        "⠼",
        "⠴",
        "⠦",
        "⠧",
        "⠇",
        "⠏"
    ]

    index = 0

    while spinner_running:

        print(
            f"\r{symbols[index % len(symbols)]} {message}",
            end="",
            flush=True
        )

        index += 1

        time.sleep(0.1)


    print(
        "\r" + " " * 70 + "\r",
        end="",
        flush=True
    )


def start_spinner(message):

    global spinner_running

    spinner_running = True

    thread = threading.Thread(
        target=spinner,
        args=(message,),
        daemon=True
    )

    thread.start()

    return thread


def stop_spinner():

    global spinner_running

    spinner_running = False


# ============================================================
# Progress bar
# ============================================================

start_time = None


def show_progress(current):

    global start_time

    if duration <= 0:
        return


    percent = min(
        100.0,
        max(
            0.0,
            (current / duration) * 100
        )
    )


    bar_length = 30


    filled = int(
        bar_length *
        percent /
        100
    )


    bar = (
        "█" * filled
        +
        "░" *
        (
            bar_length -
            filled
        )
    )


    elapsed = (
        time.time() -
        start_time
    )


    # ========================================================
    # Estimate remaining time
    # ========================================================

    if percent > 0:

        total_estimated = (
            elapsed /
            (percent / 100)
        )

        remaining = max(
            0,
            total_estimated -
            elapsed
        )

    else:

        remaining = 0


    print(
        f"\r[{bar}] "
        f"{percent:6.2f}% "
        f"{format_time(current)} / "
        f"{format_time(duration)} "
        f"| Elapsed: "
        f"{format_time(elapsed)} "
        f"| ETA: "
        f"{format_time(remaining)}",
        end="",
        flush=True
    )


# ============================================================
# Header
# ============================================================

print()

print(
    "=============================================="
)

print(
    "              VIDEO TO SRT"
)

print(
    "=============================================="
)

print()


print(
    f"Video    : "
    f"{os.path.basename(video)}"
)

print(
    f"Duration : "
    f"{format_time(duration)}"
)

print(
    f"Model    : "
    f"{MODEL_SIZE}"
)

print(
    f"Device   : "
    f"{DEVICE}"
)

print()


# ============================================================
# Load Whisper model
# ============================================================

print(
    "Loading Whisper model..."
)

print()


model_start = time.time()


spinner_thread = start_spinner(
    "Loading Whisper model..."
)


try:

    model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE
    )

finally:

    stop_spinner()

    spinner_thread.join(
        timeout=0.5
    )


model_load_time = (
    time.time() -
    model_start
)


print(
    "✓ Model loaded successfully."
)

print(
    f"  Model load time: "
    f"{format_time(model_load_time)}"
)

print()


# ============================================================
# Start transcription
# ============================================================

print(
    "Starting transcription..."
)

print()


start_time = time.time()


segments, info = model.transcribe(

    video,

    beam_size=5,

    vad_filter=True

)


# ============================================================
# Output SRT filename
# ============================================================

output = (
    os.path.splitext(video)[0]
    + ".srt"
)


# ============================================================
# SRT timestamp
# ============================================================

def timestamp(seconds):

    hours = int(
        seconds // 3600
    )


    minutes = int(
        (seconds % 3600) // 60
    )


    secs = int(
        seconds % 60
    )


    milliseconds = int(
        round(
            (
                seconds -
                int(seconds)
            ) * 1000
        )
    )


    # ========================================================
    # Handle rounding
    # ========================================================

    if milliseconds >= 1000:

        milliseconds = 0

        secs += 1


        if secs >= 60:

            secs = 0

            minutes += 1


        if minutes >= 60:

            minutes = 0

            hours += 1


    return (
        f"{hours:02}:"
        f"{minutes:02}:"
        f"{secs:02},"
        f"{milliseconds:03}"
    )


# ============================================================
# Write SRT
# ============================================================

segment_count = 0


with open(
    output,
    "w",
    encoding="utf-8"
) as f:


    for segment in segments:


        start = segment.start

        end = segment.end

        text = segment.text.strip()


        if not text:

            continue


        # Correct sequential SRT numbering

        segment_count += 1


        f.write(
            f"{segment_count}\n"
        )


        f.write(
            f"{timestamp(start)} --> "
            f"{timestamp(end)}\n"
        )


        f.write(
            f"{text}\n\n"
        )


        show_progress(end)


# ============================================================
# Finished
# ============================================================

elapsed = (
    time.time() -
    start_time
)


print()
print()


print(
    "=============================================="
)

print(
    "                 ✓ DONE!"
)

print(
    "=============================================="
)

print()


print(
    f"Video       : "
    f"{os.path.basename(video)}"
)

print(
    f"SRT         : "
    f"{os.path.basename(output)}"
)

print(
    f"Subtitles   : "
    f"{segment_count}"
)

print(
    f"Processing  : "
    f"{format_time(elapsed)}"
)

print(
    f"Language    : "
    f"{info.language}"
)

print(
    f"Probability : "
    f"{info.language_probability:.2%}"
)

print()


print(
    "SRT saved to:"
)

print(
    output
)

print()