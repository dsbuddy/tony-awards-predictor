# Video assets

Everything needed to cut a data-storytelling video about the Tony predictor.
All figures are dark-theme, 200 DPI, 16:9, generated from the live model.

## Script
- `SCRIPT.md` - full narration with visual cues and the running order.

## Core frames (narration spine)
`01_hook` `02_timeline` `03_by_year` `04_by_category` `05_calibration`
`06_forecast_2026` `07_limits`

## Supporting charts (a visual every ~10-15s)
- `08_precursor_power` - which early award is the best crystal ball
- `09_sweep_effect` - **the showpiece**: win both big awards -> 76%, neither -> 9%
- `10_agreement` - more agreement -> closer to a sure thing
- `11_model_vs_pundits` - the honest beat: pros still edge the model
- `12_climb` - accuracy as each data layer is added (58 -> 66%)
- `13_bestplay_track` - Best Play called right, year by year
- `14_scatter` - all 612 predictions at once
- `15_funnel` - coin-flip -> baseline -> model

## Deep dives
- `16_group_types` - production vs acting vs design predictability
- `17_upsets` - the famous upsets the model also missed (Avenue Q over Wicked)
- `18_bestmusical_track` - Best Musical hit/miss timeline

## Build-up sequences (for animation)
`buildup/` holds step-by-step reveal frames. Play them in order to animate a
chart building on screen:
- `09a/09b/09c` - the sweep-effect bars appearing one at a time
- `12a-12d` - the accuracy climb, point by point

## Thumbnails
- `thumb_A` - "I PREDICTED THE TONYS before they happened / 66% with math"
- `thumb_B` - "CAN MATH PREDICT THE TONY AWARDS?" with the sweep bars

## Regenerating
```bash
cd ../../src
python3 video_figures.py      # core frames 01-07
python3 video_figures2.py     # supporting 08-15
python3 video_deepdive.py     # deep dives, build-up frames, thumbnails
```
