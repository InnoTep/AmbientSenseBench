# How the generator produces events

This note documents the event-generation mechanism implemented in
`src/ambientsensebench/generate_raw_data.py`, mirroring the "Event generation"
subsection of the accompanying paper. It exists so that the artefact says the same
thing the paper says: the generator is **feature-first**, and the event layer is a
rendering of sampled daily feature values, not the trace of a simulated behavioural
process.

## The mechanism

For every simulated day, `generate_one_day`:

1. Seeds the random generator deterministically from the scenario seed, behavioural
   state, and day index (`deterministic_seed`), which is what makes the corpus exactly
   reproducible.
2. Draws the day's routine parameters **independently** from the active profile
   (`config/profiles.yaml`): wake time and sleep duration (Gaussian), meal, social,
   and night-event counts (rounded Gaussian / Poisson), target mobility and
   kitchen-activity scores (Gaussian), and an evening-consistency value (Gaussian),
   each clamped to a plausible range.
3. Calls one emitter per feature, which places exactly the events the shared extractor
   (`feature_extraction.py`) needs to recover that value, at partly stereotyped times.

In scenario mode the two profiles are interpolated by a continuous severity value in
[0, 1] that follows a sigmoid rise and fall around each episode's onset and offset
(`generate_scenarios.py`), so "severity" is a continuum, not a separate class.

## Feature-by-feature rendering

| Feature | Daily value sampled as | Rendered as |
|---|---|---|
| Wake time | N(mu, sigma), clamp 05:00-10:00 | bedroom-PIR on/off pair placed at the sampled time |
| Sleep duration | N(mu, sigma), clamp 3-12 h | bedside-lamp off event at wake time minus the sampled duration |
| Meal count | round N(mu, sigma), clamp 0-3 | kitchen-PIR and fridge open/close pairs in fixed slots at 08:00-08:40, 13:00-13:40, 19:00-19:40 |
| Social proxy | round N(mu, sigma), clamp 0-4 | front-door open/close pairs at hours 10, 15, 18, 20, filled in that order |
| Night activity | Poisson(lambda), clamp 0-12 | PIR on/off pairs at uniform times in 00:00-04:59 in a randomly chosen room |
| Mobility score | N(mu, sigma), clamp 0.1-2.0 | round(score x reference_events) non-kitchen PIR pairs (count capped at 25; fewer if the walk reaches 22:00 first) from 20 min after wake until 22:00, 30-90 min apart, each room drawn from the profile's Markov transition matrix |
| Kitchen activity score | N(mu, sigma), clamp 0.0-2.0 | extra kitchen-PIR pairs at fixed daytime hours (9-11, 15-17, 20), topping up the two events per meal to round(score x reference_events), capped at 18 |
| Evening consistency | N(mu, sigma), clamp [0, 1] | TV-off (22:00 anchor) and lamp-off (23:00 anchor) events, both shifted by (1 - value) x 4 h in one random direction |
| Room-transition entropy | not sampled directly | emerges from the interleaved room sequence of every room-mapped PIR activation (ON event) between waking and 22:00: the Markov mobility walk together with the kitchen events, which supply about 45% of that sequence on baseline days |

`reference_events` values live in `config/profiles.yaml` (mobility 9.4, kitchen 11.2).

## Consequences to keep in mind

- **Pipeline shape is parameter -> events -> feature.** The distributional validation
  is therefore a *round-trip* test of the raw-to-feature pipeline: nearly exact by
  construction for the four Gaussian features (wake time is a single event placed at
  the sampled time), genuinely informative for the count and compositional features,
  whose extractor transforms are lossy.
- **Cross-feature independence is by construction.** Each feature has its own
  independent daily sampler, so baseline cross-feature correlations are near zero
  (the largest observed, |r| = 0.17, is mobility vs. room-transition entropy, which
  share a generating mechanism: the mobility walk supplies part of the room sequence the
  entropy transform reads). Real routines couple these features; a
  behaviour-first generator with an explicit activity layer is future work.
- **The event layer has internal, not behavioural, validity.** Meals occupy fixed
  slots regardless of the sampled wake time; night activity is placed in 00:00-04:59
  independently of the sampled sleep onset; social events fill fixed hours in order.
- **Generator and extractor are arithmetically coupled.** The generator assumes the
  extractor counts two events per meal (one kitchen-PIR activation and one fridge-door
  opening)
  (`current_kitchen_raw_from_meals = meal_count * 2` in `generate_one_day`); changing
  the extractor's counting rule silently breaks the kitchen-activity targets.
  Decoupling this is planned refactor work, out of scope for the current release.
- **Regression freeze.** `tests/test_feature_matrix_freeze.py` pins the SHA-256 of the
  P01 seed-0 daily feature matrix, so any later refactor can be checked against the
  numbers reported in the paper.
