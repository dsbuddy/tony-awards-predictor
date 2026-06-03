"""
Auto-assemble the video from build_spec.json.

For each beat: synthesize voiceover (macOS `say` -> aiff -> wav), then build a
video clip that holds the beat's chart image for exactly the length of its
narration (with a small tail), then concatenate all beats into one MP4 with
fade transitions.

    python3 assemble.py            # build the whole thing
    python3 assemble.py --fast     # lower-res quick preview

Requires: ffmpeg, and macOS `say`. Output: video/tony_video.mp4

This is the "rough cut / proof of concept" path. For a polished version, hand
build_spec.json's narration to a premium TTS (ElevenLabs/Descript) and the
beat list to a real editor - see EDIT_GUIDE.md.
"""
import json
import os
import subprocess
import sys
import wave
import contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WORK = os.path.join(HERE, "_work")
os.makedirs(WORK, exist_ok=True)

FAST = "--fast" in sys.argv


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wav_seconds(path):
    with contextlib.closing(wave.open(path, "r")) as w:
        return w.getnframes() / float(w.getframerate())


PIPER_MODEL = os.path.join(HERE, "voices", "en_US-ryan-high.onnx")


def make_voice(text, voice, rate, out_wav):
    """Neural TTS via Piper if the model is present (much more human), else fall
    back to macOS `say`. length_scale > 1 slows Piper slightly for a calmer,
    more natural narration pace."""
    if os.path.exists(PIPER_MODEL):
        raw = out_wav.replace(".wav", "_raw.wav")
        p = subprocess.run(["python3", "-m", "piper", "-m", PIPER_MODEL,
                            "--length-scale", "1.05", "-f", raw],
                           input=text, text=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if p.returncode == 0 and os.path.exists(raw):
            run(["ffmpeg", "-y", "-i", raw, "-ar", "44100", "-ac", "2", out_wav])
            os.remove(raw)
            return
    # fallback: macOS say
    aiff = out_wav.replace(".wav", ".aiff")
    run(["say", "-v", voice, "-r", str(rate), "-o", aiff, text])
    run(["ffmpeg", "-y", "-i", aiff, "-ar", "44100", "-ac", "2", out_wav])
    os.remove(aiff)


def make_clip(image, audio_wav, out_mp4, dur, res):
    w, h = res.split("x")
    # scale image to fit, pad to exact resolution on the dark background
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=0x0f1117,"
          f"fade=t=in:st=0:d=0.3,fade=t=out:st={max(dur-0.3,0):.2f}:d=0.3,format=yuv420p")
    run(["ffmpeg", "-y", "-loop", "1", "-i", image, "-i", audio_wav,
         "-t", f"{dur:.2f}", "-vf", vf, "-r", "30",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
         "-shortest", out_mp4])


def main():
    spec = json.load(open(os.path.join(HERE, "build_spec.json")))
    style = spec["style"]
    res = "1280x720" if FAST else style["resolution"]
    voice = style.get("voice", "Daniel")
    rate = style.get("voice_rate_wpm", 180)

    # confirm voice exists; fall back to a common one
    avail = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    if voice not in avail:
        for alt in ("Daniel", "Alex", "Samantha", "Fred"):
            if alt in avail:
                print(f"  (voice '{voice}' not found, using '{alt}')")
                voice = alt
                break

    clips = []
    for beat in spec["beats"]:
        img = os.path.join(ROOT, beat["image"])
        if not os.path.exists(img):
            print(f"  !! missing image {beat['image']}, skipping beat {beat['id']}")
            continue
        wav = os.path.join(WORK, f"voice_{beat['id']}.wav")
        clip = os.path.join(WORK, f"clip_{beat['id']}.mp4")
        print(f"  beat {beat['id']}: voiceover...")
        make_voice(beat["narration"], voice, rate, wav)
        dur = max(wav_seconds(wav) + 0.7, beat.get("min_seconds", 4))
        print(f"  beat {beat['id']}: clip ({dur:.1f}s)...")
        make_clip(img, wav, clip, dur, res)
        clips.append(clip)

    # concat
    listfile = os.path.join(WORK, "concat.txt")
    with open(listfile, "w") as f:
        for c in clips:
            f.write(f"file '{c}'\n")
    out = os.path.join(HERE, "tony_video.mp4")
    print("  concatenating...")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile,
         "-c", "copy", out])
    # report
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", out], capture_output=True, text=True).stdout.strip()
    print(f"\nDONE -> {os.path.relpath(out)}  ({float(dur):.0f}s, {len(clips)} beats, {res})")


if __name__ == "__main__":
    main()
