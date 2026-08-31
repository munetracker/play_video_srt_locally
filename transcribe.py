from faster_whisper import WhisperModel
import sys
import os
import subprocess
import time


# ============================================================
# Configuration
# ============================================================

MODEL_SIZE = "base"
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
    print(f"ERROR: File not found:")
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

        return float(result.stdout.strip())

    except Exception as e:

        print()
        print("ERROR: Could not determine video duration.")
        print(e)
        print()

        sys.exit(1)


duration = get_duration(video)


# ============================================================
# Format time
# ============================================================

def format_time(seconds):

    seconds = max(0, int(seconds))

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    return f"{hours:02}:{minutes:02}:{secs:02}"


# ============================================================
# Progress bar
# ============================================================

start_time = None


def show_progress(current):

    global start_time

    if duration <= 0:
        return

    percent = min(100.0, max(0.0, (current / duration) * 100))

    bar_length = 30

    filled = int(bar_length * percent / 100)

    bar = (
        "█" * filled
        + "░" * (bar_length - filled)
    )

    elapsed = time.time() - start_time

    # Estimate remaining time
    if percent > 0:
        total_estimated = elapsed / (percent / 100)
        remaining = max(0, total_estimated - elapsed)
    else:
        remaining = 0

    print(
        f"\r[{bar}] "
        f"{percent:6.2f}% "
        f"{format_time(current)} / {format_time(duration)} "
        f"| Elapsed: {format_time(elapsed)} "
        f"| ETA: {format_time(remaining)}",
        end="",
        flush=True
    )


# ============================================================
# Header
# ============================================================

print()
print("==============================================")
print("              VIDEO TO SRT")
print("==============================================")
print()

print(f"Video    : {os.path.basename(video)}")
print(f"Duration : {format_time(duration)}")
print(f"Model    : {MODEL_SIZE}")
print(f"Device   : {DEVICE}")
print()

# ============================================================
# Load Whisper
# ============================================================

print("Loading Whisper model...")
print()

model = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE
)

print("Model loaded.")
print()
print("Starting transcription...")
print()


# ============================================================
# Start transcription
# ============================================================

start_time = time.time()

segments, info = model.transcribe(
    video,
    beam_size=5,
    vad_filter=True
)


# ============================================================
# Output SRT filename
# ============================================================

output = os.path.splitext(video)[0] + ".srt"


# ============================================================
# SRT timestamp
# ============================================================

def timestamp(seconds):

    hours = int(seconds // 3600)

    minutes = int(
        (seconds % 3600) // 60
    )

    secs = int(seconds % 60)

    milliseconds = int(
        round(
            (seconds - int(seconds)) * 1000
        )
    )

    # Handle rounding to 1000 ms
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

    for segment_count, segment in enumerate(
        segments,
        start=1
    ):

        start = segment.start
        end = segment.end

        text = segment.text.strip()

        if not text:
            continue

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

elapsed = time.time() - start_time

print()
print()

print("==============================================")
print("                 DONE!")
print("==============================================")

print()

print(f"Video       : {os.path.basename(video)}")
print(f"SRT         : {os.path.basename(output)}")
print(f"Subtitles   : {segment_count}")
print(f"Processing  : {format_time(elapsed)}")

print()

print("SRT saved to:")

print(output)

print()