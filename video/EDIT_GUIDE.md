# Video production guide

Two ways to make the video. The rough cut is already built; the polished path
is below.

## What's here

- `build_spec.json` - the single source of truth. Every beat: chart image,
  narration text, minimum duration, transition. Edit this and re-run to change
  anything.
- `assemble.py` - auto-builds the rough-cut MP4 from the spec (macOS `say`
  voiceover + ffmpeg). Run `python3 assemble.py` (or `--fast` for 720p preview).
- `tony_video.mp4` - the auto-assembled rough cut (~4 min, 1080p).
- `narration.txt` - the full script as plain text, ready to paste into a premium
  TTS or to read yourself.

## Path A: the auto-built cut (done)

Already built, with a **neural voice** (Piper). Good for a quick share or to
sanity-check pacing - the narration sounds genuinely human, and the timing,
transitions, and chart sync are all real.

The voice model (~115 MB) isn't committed; see `voices/README.md` to fetch it
(one `curl`). If it's absent the assembler falls back to the robotic macOS
`say` voice. To rebuild after editing narration: `python3 assemble.py`

For an even better voice, see Path B (ElevenLabs/Descript or your own recording).

## Path B: the polished version

1. **Better voice.** Paste `narration.txt` into ElevenLabs or Descript, pick a
   warm conversational voice, export the audio. (Or record yourself - most
   authentic for an "I built this" video.)
2. **Assemble in a real editor** (CapCut, Descript, Premiere, Resolve):
   - Drop the chart PNGs (`../output/video/*.png`) on the timeline in the order
     listed in `build_spec.json` (it matches SCRIPT.md's running order).
   - Lay the voiceover under them; trim each chart to its narration.
   - For beats with a `buildup` array (sweep 09, climb 12), use the
     `../output/video/buildup/` frames in sequence to animate the chart building
     in - hold each frame ~0.4s.
   - Add the Veo B-roll (below) as 2-4s cutaways at the cold open and between
     sections.
3. **Thumbnail:** `../output/video/thumb_A.png` or `thumb_B.png`.
4. **Music:** something building under the data sections, drop out for the
   "what it can't do" close.

## Veo / AI B-roll prompts

`build_spec.json` -> `veo_broll_prompts` has 5 ready-to-paste prompts for
cinematic theatre B-roll (marquee, Times Square, empty auditorium, trophy,
Playbill). These are flavor cutaways - the charts carry the story, so B-roll is
optional polish.

## Asking an AI tool to assemble it

No single tool (Gemini, Veo) ingests your charts + script and outputs a finished
video today. The realistic AI-assisted flow:
- **Veo** -> generate the B-roll clips from the prompts above.
- **ElevenLabs/Descript** -> generate the voiceover from `narration.txt`.
- **Descript or CapCut** -> drop in the charts + voiceover + B-roll; both have
  AI helpers that auto-trim images to narration.
- `build_spec.json` is structured so you (or a script) can feed any of these
  tools beat-by-beat without re-deriving timings.
