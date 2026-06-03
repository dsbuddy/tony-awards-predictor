# Can You Predict the Tony Awards Before They Happen?
### Video script (data-storytelling style)

Target length: ~6-8 minutes. Tone: curious, conversational, a little dramatic.
Narration is plain text. `[VISUAL: ...]` cues map each beat to a figure in this
folder or to suggested B-roll. Read at a relaxed pace; let the reveals breathe.

---

## COLD OPEN (the hook) - ~0:00

[VISUAL: 01_hook.png - the big "66%" on black, hold on it]

> Every June, a few hundred people decide the biggest awards in American
> theatre. The Tonys. And every June, theatre fans argue for weeks about who's
> going to win.
>
> But here's the thing nobody really says out loud: by the time the Tonys
> happen, a *lot* of the answer is already sitting there in plain sight. In the
> awards that came before.
>
> I built a model that only looks at those earlier awards. No insider info, no
> crystal ball. And it calls the Tony winners correctly about two-thirds of the
> time. Sixty-six percent. Across every category, going back to the year 2000.
>
> So today I want to answer a question I've been chewing on for years: *how
> predictable are the Tonys, really?* And then I'm going to use it to make my
> picks for this year, live, before the ceremony. So you can watch me be right.
> Or watch me eat it.

[beat]

---

## ACT 1 - THE SETUP (why this should even be possible) - ~0:50

[VISUAL: 02_timeline.png - the precursor-awards timeline building left to right]

> Here's the setup. The Tonys are the *last* award show of the Broadway season,
> not the first. Before they happen, you get the Drama Desk Awards. The Outer
> Critics Circle. The Drama League. The New York Drama Critics' Circle.
>
> Four separate groups, all voting on the same shows, all announcing their
> winners in the weeks *before* the Tonys. And year after year, the same names
> keep coming up. A show sweeps the warm-up awards, and then... it wins the
> Tony too.
>
> Film nerds have done this forever with the Oscars. If a movie wins the
> Directors Guild, the Producers Guild, and the SAG ensemble award, it's
> basically already won Best Picture. Theatre just doesn't get the same
> treatment. So I wanted to actually measure it.

> The catch - and this is the whole game - I only let the model use information
> that existed *before* the ceremony. The moment you let it peek at anything
> from after, you're not predicting anymore, you're cheating. I'll come back to
> how seriously I took that, because it almost broke the whole thing.

---

## ACT 2 - DOES IT ACTUALLY WORK? (the proof) - ~1:50

[VISUAL: 03_by_year.png - accuracy by season vs baseline]

> So does it work? Here's the honest test. I hide one year completely. Train the
> model on every *other* year. Then ask it to predict the year it never saw. And
> I do that for all twenty-six seasons.
>
> The gold bars are the model. That red line is what you'd get from a dumb
> baseline - just pick whoever won the most earlier awards. The model beats it
> almost every single year, and lands at about sixty-six percent overall.
>
> Now, sixty-six percent might not sound jaw-dropping. But most of these
> categories have five nominees. Random guessing gets you twenty percent. So
> we're more than three times better than a coin flip - and a solid step above
> the obvious baseline.

[beat]

> But the average hides the most interesting part. Because the model is *not*
> equally good at everything.

[VISUAL: 04_by_category.png - accuracy by category, green to red]

> Look at this. Best Play? The earlier awards basically *solve* it - ninety-six
> percent. Best Actor in a Play, the revivals, directing - all way up here in
> lock territory.
>
> And then... the bottom. Sound design. Some of the featured acting races. Down
> here it's barely better than flipping a coin. Why? Because in those categories
> the earlier-award groups genuinely *disagree* with each other - and with the
> Tony voters. There's no signal to find. The model knows it doesn't know.
>
> And honestly? That's the part I'm proudest of. A good model isn't one that's
> always confident. It's one that's confident when it should be, and humble when
> it shouldn't.

---

## ACT 3 - CAN YOU TRUST THE CONFIDENCE? (the credibility check) - ~3:20

[VISUAL: 05_calibration.png - calibration scatter vs the diagonal]

> Which brings up the real test of any prediction model. Not "is it right" - but
> "when it says it's seventy percent sure, is it *actually* seventy percent
> sure?"
>
> This chart answers that. The dotted line is perfect honesty. Every dot is the
> model saying "I'm this confident" - and where the dot lands tells you how
> often it was actually right.
>
> See how the dots hug the line? When this model says something's a lock, it
> wins about as often as it claims. It's not blowing smoke. That's the
> difference between a model and a guy on Twitter yelling predictions.

[optional aside]

> I'll be straight with you - the raw model was a little *over*confident at
> first, always wanting to scream a hundred percent. So I added a correction
> layer that pulls those numbers back to what the history actually supports. The
> result is the honest version you're seeing.

---

## ACT 4 - THE PAYOFF (this year's picks) - ~4:20

[VISUAL: 06_forecast_2026.png - the ranked 2026 forecast, hold]

> Okay. Enough history. Here's what you came for. This year's picks - made
> before the ceremony, ranked by how confident the model is.
>
> Green is a lock. Yellow is a lean. Red is the model throwing up its hands and
> saying "genuinely don't know."
>
> The safe money: [name your top locks from the chart - e.g. Death of a Salesman
> sweeping the play and revival side, Ragtime on the musical revival side]. The
> earlier awards lined up almost perfectly behind these, and history says that's
> a near-guarantee.
>
> The toss-ups - [name the red ones, e.g. the featured races, sound design] -
> those are the ones to actually watch on the night. If there's an upset, it's
> coming from here. That's where your bracket gets busted.

[beat - if you want a "Curry moment" / single dramatic callout, put it here:]

> And there's one pick I want to flag, because it's the one the model and the
> internet disagree on... [insert the most interesting divergence - a category
> where the human consensus is a lock but the model hedges, or vice versa]. We'll
> know in a few days who was right.

---

## ACT 5 - THE HONEST CLOSE (what it can't do) - ~5:40

[VISUAL: 07_limits.png - "What the model CAN'T see"]

> Now before you go bet your rent on this - here's what this thing *cannot* do.
>
> It reads the earlier awards. It does not read the *room*. It can't feel a
> career-defining performance that brings the house down. It can't sense when
> two similar shows are about to split the vote and hand it to a third. And some
> years - the weird ones, the wide-open ones - the data just genuinely doesn't
> know. Same as the rest of us.
>
> The Tonys, like any award show, don't always reward the thing everyone admires
> most. That gap - between what the numbers predict and what actually happens -
> that's not a bug in the model. That's the whole reason we still watch.
>
> So those are my picks. The code, the data, all of it is public - link's in the
> description if you want to tear it apart or build your own. Tell me in the
> comments who you think it's got wrong. And I'll see you after the ceremony to
> find out how dumb I look.

[VISUAL: back to 01_hook.png or a "we'll find out Sunday" card]

---

## EXPANDED VISUAL MAP (drop these in for a chart every ~10-15s)

The 7 core frames above carry the spine. These 8 extra charts (08-15) layer in
so there's always something moving on screen. Suggested placement:

- **08_precursor_power** - in ACT 1, when you introduce the four awards: "and
  some of these are far better crystal balls than others." Reveal the ranking.
- **09_sweep_effect** - the showpiece. Drop it in ACT 2 right after the headline
  number: "here's the single most useful fact in the whole project." Win both
  Drama Desk + Outer Critics -> 76% Tony win. Win neither -> 9%. Let it land.
- **10_agreement** - immediately after sweep: "and the more of them that agree,
  the more it becomes a formality." The line climbs to ~100%.
- **12_climb** - in ACT 2 transition: "so I kept stacking on everything else you
  could know beforehand, and watched the accuracy climb." 58 -> 61 -> 63 -> 66.
- **11_model_vs_pundits** - the humility beat, late ACT 3 or into ACT 5: "could a
  model beat the human experts? Honestly... on the races they bother to call,
  no. The pros still edge it." Builds huge credibility by admitting it.
- **14_scatter** - in ACT 3 with calibration: "every single prediction, 26
  seasons at once - notice the misses cluster on the low-confidence side."
- **13_bestplay_track** - in ACT 2 when you hit the "solved" categories: show
  Best Play called right year after year, a row of green.
- **15_funnel** - optional ACT 2 or recap: the clean "coin flip -> baseline ->
  model" three-tier visual for a quick mid-roll re-hook.

So a fuller running order is roughly:
01 hook -> 02 timeline -> 08 power -> 09 sweep -> 10 agreement -> 12 climb ->
03 by-year -> 13 best-play -> 04 by-category -> 15 funnel -> 05 calibration ->
14 scatter -> 11 model-vs-pundits -> 06 forecast -> 07 limits.

## PRODUCTION NOTES

- **Figures** are in this folder, numbered in narration order. All 1920-friendly,
  200 DPI, dark theme - they'll screen-record clean or drop into any editor.
- **Fill in the brackets** in Act 4 with the specific 2026 picks from the
  forecast chart / PREDICTIONS_2026.md. I left them as cues so the script stays
  accurate if you regenerate predictions.
- **Pacing:** the reveals (the 66%, the category chart, the forecast) want a beat
  of silence after them. Don't rush the gold number.
- **The "Curry moment"** (a single surprising trace-it-back story) is the one
  thing this script gestures at but can't fully write for you - if there's a
  category where one earlier award flipped the whole prediction, that's your
  most shareable 30 seconds. Worth digging for.
- **Music:** something building under Act 2-4, drop out for the limits section.
