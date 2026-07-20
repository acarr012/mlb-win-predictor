"""
data_loader.py

Handles raw data acquisition from pybaseball:
- Team schedule/record data (Baseball-Reference, via schedule_and_record)
- Statcast pitch-level data (added in a later step)
"""

import time
import numpy as np
import pandas as pd
from pybaseball import schedule_and_record
import pybaseball.team_results as _team_results

# Workaround for two bugs in pybaseball 2.2.7's team_results.get_table():
#
# 1. It uses a chained `df['Attendance'].replace(..., inplace=True)`, which
#    silently no-ops under pandas' Copy-on-Write (mandatory as of pandas 3.x).
#    That leaves the literal string "Unknown" in the Attendance column, which
#    later blows up when make_numeric() tries to cast it to float.
#
# 2. It iterates `range(len(rows) - 1)`, deliberately dropping the LAST <tr>
#    in the schedule table on the assumption it's always a trailing "column
#    legend" footer row. On the current Baseball-Reference page format
#    there's no such footer — verified by inspecting the raw HTML for
#    NYY 2024 (168 <tr> = 162 real games + 6 repeated mid-table header
#    rows, no legend row) — so this silently drops the final game of any
#    completed season. The existing try/except already skips the repeated
#    header rows correctly (they're too short to index into), so we just
#    need to stop pre-emptively excluding the last row.
#
# Reimplemented here (rather than patched in place) because bug #2 lives
# inside the HTML-parsing loop itself, not something fixable on the
# returned DataFrame.
def _patched_get_table(soup, team):
    try:
        table = soup.find_all("table")[0]
    except Exception:
        raise ValueError(
            "Data cannot be retrieved for this team/year combo. Please verify "
            "that your team abbreviation is accurate and that the team existed "
            "during the season you are searching for."
        )
    data = []
    headings = [th.get_text() for th in table.find("tr").find_all("th")]
    headings = headings[1:]  # the "gm#" heading doesn't have a <td> element
    headings[3] = "Home_Away"
    data.append(headings)
    table_body = table.find("tbody")
    rows = table_body.find_all("tr")
    for row_index in range(len(rows)):  # fixed: no longer drops the last row
        row = rows[row_index]
        try:
            cols = row.find_all("td")
            if cols[1].text == "":
                cols[1].string = team
            if cols[3].text == "":
                cols[3].string = "Home"
            if cols[12].text == "":
                cols[12].string = "None"
            if cols[13].text == "":
                cols[13].string = "None"
            if cols[14].text == "":
                cols[14].string = "None"
            if cols[8].text == "":
                cols[8].string = "9"
            if cols[16].text == "":
                cols[16].string = "Unknown"
            if cols[15].text == "":
                cols[15].string = "Unknown"
            if cols[17].text == "":
                cols[17].string = "Unknown"

            cols = [ele.text.strip() for ele in cols]
            data.append([ele for ele in cols if ele])
        except Exception:
            if len(cols) > 1:
                cols = [ele.text.strip() for ele in cols][0:5]
                data.append([ele for ele in cols if ele])

    df = pd.DataFrame(data)
    df = df.rename(columns=df.iloc[0])
    df = df.reindex(df.index.drop(0))
    df = df.drop("", axis=1)
    df["Attendance"] = df["Attendance"].replace(r"^Unknown$", np.nan, regex=True)
    return df


_team_results.get_table = _patched_get_table


def pull_schedule_data(teams, seasons, sleep_seconds=1):
    """
    Pulls game-by-game schedule/record data for the given teams and seasons.

    Each game will appear TWICE in the raw output — once from each team's
    perspective — because we're pulling per-team schedules. This is expected
    and gets resolved later during the home-row dedup step, not here.

    Parameters
    ----------
    teams : list of str
        Baseball-Reference team abbreviations (e.g., 'NYY', 'BOS')
    seasons : list of int
        Seasons to pull (e.g., [2024])
    sleep_seconds : int
        Delay between requests to avoid rate-limiting Baseball-Reference

    Returns
    -------
    pd.DataFrame
        Raw concatenated schedule data, one row per team per game
    """
    all_games = []

    for season in seasons:
        for team in teams:
            try:
                df = schedule_and_record(season, team)
                df['team'] = team
                df['season'] = season
                all_games.append(df)
                print(f"Pulled {team} {season}: {len(df)} rows")
            except Exception as e:
                print(f"FAILED: {team} {season} — {e}")

            time.sleep(sleep_seconds)

    return pd.concat(all_games, ignore_index=True)


def clean_schedule_data(raw_df):
    """
    Converts raw two-rows-per-game schedule data into one row per game.

    Uses only the home-team rows as the source of truth — this IS the dedup
    step. We never touch the away-team copy of each game, so there's no
    double counting.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Output of pull_schedule_data()

    Returns
    -------
    pd.DataFrame
        One row per game: date, home_team, away_team, home_score,
        away_score, home_team_won
    """
    df = raw_df.copy()

    # Keep only home-team rows — this is the entire dedup mechanism
    home_rows = df[df['Home_Away'] == 'Home'].copy()

    home_rows = home_rows.rename(columns={
        'team': 'home_team',
        'Opp': 'away_team',
        'R': 'home_score',
        'RA': 'away_score',
    })

    # Doubleheader games are suffixed "Aug 7 (1)" / "Aug 7 (2)" on
    # Baseball-Reference. Both games share the same calendar date, so we
    # pull the "(1)"/"(2)" marker into its own column BEFORE parsing the
    # date — otherwise game 1 and game 2 collapse into a single row.
    # Single games have no marker, so they default to game_number 1.
    game_number = (
        home_rows['Date'].str.extract(r'\((\d+)\)$')[0]
        .fillna(1)
        .astype(int)
    )
    home_rows['game_number'] = game_number

    # Parse the "Thursday, Mar 28" style date, using the season column
    # we already attached during the pull, since the raw date has no year
    home_rows['date'] = pd.to_datetime(
        home_rows['Date'].str.replace(r'^\w+,\s*', '', regex=True)
        .str.replace(r'\s*\(\d+\)$', '', regex=True)
        + ' '
        + home_rows['season'].astype(str),
        format='%b %d %Y'
    )

    # Drop ties/suspended games explicitly — don't let them silently
    # become home_team_won = 0, which would be factually wrong
    home_rows = home_rows[home_rows['home_score'] != home_rows['away_score']]

    home_rows['home_team_won'] = (
        home_rows['home_score'] > home_rows['away_score']
    ).astype(int)

    return home_rows[[
        'date', 'game_number', 'season', 'home_team', 'away_team',
        'home_score', 'away_score', 'home_team_won'
    ]].sort_values(['date', 'game_number']).reset_index(drop=True)


if __name__ == "__main__":
    test_teams = ['NYY', 'BOS']
    test_seasons = [2024]

    raw = pull_schedule_data(test_teams, test_seasons)

    print("\n--- Home_Away column raw values ---")
    print(raw['Home_Away'].unique())

    cleaned = clean_schedule_data(raw)

    print("\n--- Cleaned game table ---")
    print(cleaned.head(10))
    print(f"\nTotal games: {len(cleaned)}")
    print(f"Raw rows before dedup: {len(raw)}")

