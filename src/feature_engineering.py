"""
feature_engineering.py

Derives predictive features from cleaned raw data:
- Per-pitcher-per-game stat lines (from Statcast pitch-level events)
- Rolling team and pitcher statistics (to be added)
"""

import pandas as pd

# Maps each Statcast `events` value to how many outs that specific play
# recorded. This is the foundation for innings pitched (outs / 3), which
# ERA, WHIP, and FIP all depend on.
#
# Judgment calls worth flagging explicitly:
# - sac_fly / sac_bunt: batter is out, so these count as 1 out, even
#   though they don't count as an "at-bat" in traditional batting stats.
#   For OUTS RECORDED (a pitching stat), that distinction doesn't matter.
# - fielders_choice: the batter reaches base safely, but a baserunner is
#   put out elsewhere on the play. Counts as 1 out for the pitcher's line.
# - field_error: batter reaches on a defensive error — no out recorded,
#   and NOT the pitcher's fault, but Statcast's `events` field doesn't
#   separate "error" from other on-base outcomes any further than this.
# - catcher_interf: batter awarded first base due to catcher interference.
#   Not an out, not really a "walk" either — excluded from both outs and
#   walk counts since it's not something the pitcher controls at all.
# - truncated_pa: an incomplete plate appearance (game suspended/ended
#   mid-AB). Excluded entirely — there's no real outcome to classify.
OUTS_PER_EVENT = {
    'field_out': 1,
    'strikeout': 1,
    'force_out': 1,
    'sac_fly': 1,
    'sac_bunt': 1,
    'fielders_choice_out': 1,
    'grounded_into_double_play': 2,
    'double_play': 2,
    'strikeout_double_play': 2,
    'sac_fly_double_play': 2,
    'triple_play': 3,
    # Everything else (single, double, triple, home_run, walk,
    # intent_walk, hit_by_pitch, field_error, fielders_choice,
    # catcher_interf, truncated_pa) records 0 outs.
}

# Events that count as a "hit" for WHIP purposes
HIT_EVENTS = {'single', 'double', 'triple', 'home_run'}

# Events that count as a "walk" for WHIP purposes (intentional walks
# still put a runner on base, so they count same as a regular walk)
WALK_EVENTS = {'walk', 'intent_walk'}


def build_pitcher_game_log(statcast_df):
    """
    Aggregates pitch-level Statcast data into one row per (pitcher, game),
    with the counting stats needed to derive ERA, WHIP, and FIP later.

    Parameters
    ----------
    statcast_df : pd.DataFrame
        Cleaned, regular-season-only Statcast data (output of
        clean_statcast_data() in data_loader.py)

    Returns
    -------
    pd.DataFrame
        One row per (game_pk, pitcher), with outs_recorded, hits_allowed,
        walks_allowed, home_runs_allowed, strikeouts, hit_by_pitch
    """
    # Only rows where a plate appearance actually ended have a real
    # events value — most pitches (balls, called strikes mid-at-bat)
    # have events == NaN and aren't outcomes on their own.
    outcomes = statcast_df[statcast_df['events'].notna()].copy()

    outcomes['outs_this_play'] = outcomes['events'].map(OUTS_PER_EVENT).fillna(0)
    outcomes['is_hit'] = outcomes['events'].isin(HIT_EVENTS)
    outcomes['is_walk'] = outcomes['events'].isin(WALK_EVENTS)
    outcomes['is_home_run'] = outcomes['events'] == 'home_run'
    outcomes['is_strikeout'] = outcomes['events'].isin(
        ['strikeout', 'strikeout_double_play']
    )
    outcomes['is_hbp'] = outcomes['events'] == 'hit_by_pitch'

    game_log = outcomes.groupby(['game_pk', 'pitcher']).agg(
        outs_recorded=('outs_this_play', 'sum'),
        hits_allowed=('is_hit', 'sum'),
        walks_allowed=('is_walk', 'sum'),
        home_runs_allowed=('is_home_run', 'sum'),
        strikeouts=('is_strikeout', 'sum'),
        hit_by_pitch=('is_hbp', 'sum'),
        game_date=('game_date', 'first'),
        season=('game_year', 'first'),
    ).reset_index()

    return game_log



def add_rolling_pitcher_stats(game_log, window=None):
    """
    Computes rolling pre-game WHIP and FIP for each pitcher, using only
    starts strictly before the current game (shift(1), same leakage
    discipline as the team-level rolling features).

    Parameters
    ----------
    game_log : pd.DataFrame
        Output of build_pitcher_game_log(), with a 'season' column
        already attached (merge this in before calling, if not present)
    window : int or None
        Number of most recent starts to include. None = cumulative,
        season-to-date (all starts so far this season). An integer
        (e.g. 3) = rolling window over just the last N starts.

    Returns
    -------
    pd.DataFrame
        game_log with two new columns added: whip_entering_game and
        fip_entering_game (column names include the window size if given)
    """
    game_log = game_log.sort_values(['pitcher', 'season', 'game_date']).copy()
    grouped = game_log.groupby(['pitcher', 'season'])

    if window is None:
        outs = grouped['outs_recorded'].transform(lambda x: x.shift(1).expanding().sum())
        hits = grouped['hits_allowed'].transform(lambda x: x.shift(1).expanding().sum())
        walks = grouped['walks_allowed'].transform(lambda x: x.shift(1).expanding().sum())
        hr = grouped['home_runs_allowed'].transform(lambda x: x.shift(1).expanding().sum())
        k = grouped['strikeouts'].transform(lambda x: x.shift(1).expanding().sum())
        hbp = grouped['hit_by_pitch'].transform(lambda x: x.shift(1).expanding().sum())
        suffix = 'season'
    else:
        outs = grouped['outs_recorded'].transform(lambda x: x.shift(1).rolling(window).sum())
        hits = grouped['hits_allowed'].transform(lambda x: x.shift(1).rolling(window).sum())
        walks = grouped['walks_allowed'].transform(lambda x: x.shift(1).rolling(window).sum())
        hr = grouped['home_runs_allowed'].transform(lambda x: x.shift(1).rolling(window).sum())
        k = grouped['strikeouts'].transform(lambda x: x.shift(1).rolling(window).sum())
        hbp = grouped['hit_by_pitch'].transform(lambda x: x.shift(1).rolling(window).sum())
        suffix = f'last{window}'

    innings_pitched = outs / 3

    game_log[f'whip_entering_game_{suffix}'] = (walks + hits) / innings_pitched

    # FIP formula: standard constant (~3.10, varies slightly by year/league)
    # added so FIP sits on the same numeric scale as ERA for interpretability
    fip_constant = 3.10
    game_log[f'fip_entering_game_{suffix}'] = (
        (13 * hr + 3 * (walks + hbp) - 2 * k) / innings_pitched
    ) + fip_constant

    return game_log