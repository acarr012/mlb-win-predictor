"""
feature_engineering.py

Derives predictive features from cleaned raw data:
- Per-pitcher-per-game stat lines (from Statcast pitch-level events)
- Rolling team and pitcher statistics (to be added)
"""

import pandas as pd
import os
from data_loader import pull_statcast_season, clean_statcast_data, identify_starting_pitchers

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



if __name__ == "__main__":
    os.makedirs('data/processed', exist_ok=True)

    seasons = [2021, 2022, 2023, 2024, 2025, 2026]
    all_game_logs = []

    for season in seasons:
        raw_statcast = pull_statcast_season(season)
        regular_season = clean_statcast_data(raw_statcast)

        game_log = build_pitcher_game_log(regular_season)
        game_log = add_rolling_pitcher_stats(game_log, window=None)  # season-to-date
        game_log = add_rolling_pitcher_stats(game_log, window=3)     # last 3 starts
        game_log = add_rolling_pitcher_stats(game_log, window=5)     # last 5 starts

        game_log.to_parquet(f'data/processed/pitcher_game_log_{season}.parquet')
        all_game_logs.append(game_log)

        print(f"Season {season}: {len(game_log)} pitcher-game rows\n")

    master_pitcher_log = pd.concat(all_game_logs, ignore_index=True)
    master_pitcher_log.to_parquet('data/processed/master_pitcher_game_log.parquet')

    print(f"\nTotal pitcher-game rows across all seasons: {len(master_pitcher_log)}")
    print(master_pitcher_log.groupby('season').size())



import glob

# Statcast uses different team abbreviations than Baseball-Reference
# (schedule data) for 7 franchises. Maps Statcast's code -> the schedule
# table's code, so downstream merges match correctly instead of silently
# dropping ~a quarter of a season's games.
STATCAST_TO_SCHEDULE_ABBR = {
    'AZ': 'ARI',
    'CWS': 'CHW',
    'KC': 'KCR',
    'SD': 'SDP',
    'SF': 'SFG',
    'TB': 'TBR',
    'WSH': 'WSN',
}

# The Athletics are a special case: Statcast always reports them as "ATH",
# but the schedule table's abbreviation depends on the season it was
# pulled in. Seasons 2021-2024 were cached under Baseball-Reference's old
# "OAK" abbreviation (correct at the time); 2025+ was pulled fresh under
# "ATH" after the franchise's move/rename. So this one mapping flips
# depending on which season we're building.
ATHLETICS_RENAME_SEASON = 2025


def build_starters_with_game_info(season):
    """
    Reloads cached raw Statcast monthly files for a season, filters to
    regular season, identifies starters, and attaches each game's date
    and teams (needed to merge against the schedule table later, which
    has no game_pk of its own).

    Team abbreviations are normalized to match the schedule table's
    convention here, at the source — so every downstream use of this
    function's output is automatically safe, rather than relying on
    whoever calls it to remember to apply the crosswalk separately.
    """
    monthly_files = glob.glob(f'data/raw/statcast_{season}_*.parquet')
    raw = pd.concat([pd.read_parquet(f) for f in monthly_files], ignore_index=True)
    regular_season = clean_statcast_data(raw)

    starters = identify_starting_pitchers(regular_season)

    game_info = regular_season.groupby('game_pk').agg(
        game_date=('game_date', 'first'),
        home_team=('home_team', 'first'),
        away_team=('away_team', 'first'),
    ).reset_index()

    result = starters.merge(game_info, on='game_pk')

    abbr_map = dict(STATCAST_TO_SCHEDULE_ABBR)
    if season < ATHLETICS_RENAME_SEASON:
        abbr_map['ATH'] = 'OAK'

    result['home_team'] = result['home_team'].replace(abbr_map)
    result['away_team'] = result['away_team'].replace(abbr_map)

    # Doubleheader games share the same (date, home_team, away_team), so
    # without a tiebreaker a merge against the schedule table's
    # game_number column would match each schedule row against BOTH
    # games instead of just its own — silently doubling and cross-wiring
    # starter assignments for every doubleheader date. game_pk is
    # assigned in play order, so ranking within the group recovers which
    # game was 1 and which was 2.
    result['game_number'] = (
        result.groupby(['home_team', 'away_team', 'game_date'])['game_pk']
        .rank(method='first')
        .astype(int)
    )

    # Statcast's game_date comes through as a plain string; the schedule
    # table's date column is datetime64. Normalize here so merges against
    # the schedule table work out of the box instead of raising a dtype
    # ValueError on every caller.
    result['game_date'] = pd.to_datetime(result['game_date'])

    return result