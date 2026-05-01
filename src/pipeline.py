#!/usr/bin/env python3
"""Final World Cup modeling pipeline.


    python3 run_final_pipeline.py
"""
from __future__ import annotations

import json
import math
import os
import pickle
import random
import sys
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import combinations, product
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

FINAL_DIR = Path(__file__).resolve().parents[1]
OUT = FINAL_DIR / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
MODELS_DIR = OUT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
FORECAST_DIR = OUT / "forecast_variants"
FORECAST_DIR.mkdir(parents=True, exist_ok=True)

RAW_MATCHES = FINAL_DIR / "data" / "raw_matches.pkl"
SUPPLEMENTAL_2026_RESULTS = FINAL_DIR / "data" / "international_results_2022_2026.csv"
GROUPS_2026 = FINAL_DIR / "config" / "2026_groups.csv"

LABELS = [0, 1, 2]
LABEL_STR = {0: "Win", 1: "Draw", 2: "Loss"}
KNOCKOUT_STAGES = {"R32", "R16", "QF", "SF", "3rd", "F", "Champion"}
MIN_TUNING_TRAIN_ROWS = 35
TEST_YEAR = 2022
MAX_GOALS = 11
RNG_SEED = 42
HOSTS_2026 = {"Mexico", "Canada", "United States"}

ORIGINAL_18_FEATURES = [
    "elo_a", "elo_b", "wr_a", "wr_b", "gd_a", "gd_b",
    "gs_a", "gs_b", "opp_elo_a", "opp_elo_b", "cs_a", "cs_b",
    "wc_hist_a", "wc_hist_b", "h2h_a", "stage_ord", "upset_a", "gk_b",
]

FEATURE_GLOSSARY = {
    "elo_a": ("Strength", "Pre-match Elo rating of the higher-rated team A."),
    "elo_b": ("Strength", "Pre-match Elo rating of the lower-rated team B."),
    "wr_a": ("Recent form", "Team A win-rate over the recent-match form window."),
    "wr_b": ("Recent form", "Team B win-rate over the recent-match form window."),
    "gd_a": ("Recent form", "Team A average goal differential over the form window."),
    "gd_b": ("Recent form", "Team B average goal differential over the form window."),
    "gs_a": ("Recent attack", "Team A average goals scored over the form window."),
    "gs_b": ("Recent attack", "Team B average goals scored over the form window."),
    "opp_elo_a": ("Schedule strength", "Average recent opponent Elo faced by team A."),
    "opp_elo_b": ("Schedule strength", "Average recent opponent Elo faced by team B."),
    "cs_a": ("Recent defense", "Team A clean-sheet rate over the form window."),
    "cs_b": ("Recent defense", "Team B clean-sheet rate over the form window."),
    "wc_hist_a": ("Tournament history", "Time-decayed deepest prior World Cup stage reached by team A."),
    "wc_hist_b": ("Tournament history", "Time-decayed deepest prior World Cup stage reached by team B."),
    "h2h_a": ("Matchup", "Historical head-to-head score for team A against team B."),
    "stage_ord": ("Context", "Ordinal match-stage indicator from group stage through final."),
    "upset_a": ("Upset tendency", "EMA rate at which team A lost matches while favored by Elo."),
    "gk_b": ("Underdog tendency", "EMA rate at which team B won matches while unfavored by Elo."),
}

STAGE_ORD = {
    "international": -1,
    "group": 0,
    "R16": 1,
    "R32": 1,
    "QF": 2,
    "SF": 3,
    "3rd": 3,
    "F": 4,
}

TEAM_ALIASES = {
    "Czechia": "Czech Republic",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "USA": "United States",
    "Cote d'Ivoire": "Cote d'Ivoire",
    "Côte d'Ivoire": "Cote d'Ivoire",
    "Ivory Coast": "Cote d'Ivoire",
    "Curacao": "Curacao",
    "Curaçao": "Curacao",
    "Cabo Verde": "Cape Verde",
    "Cape Verde Islands": "Cape Verde",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Bosnia-Herzegovina": "Bosnia-Herzegovina",
    "Turkiye": "Turkey",
    "Türkiye": "Turkey",
    "Congo DR": "DR Congo",
    "DR Congo": "DR Congo",
    "Korea DPR": "North Korea",
    "China PR": "China",
}


def norm_team(name: str) -> str:
    name = str(name).strip()
    return TEAM_ALIASES.get(name, name)


def display_team(name: str) -> str:
    reverse = {
        "Cape Verde": "Cabo Verde",
        "Turkey": "Turkiye",
        "DR Congo": "Congo DR",
        "Cote d'Ivoire": "Cote d'Ivoire",
    }
    return reverse.get(name, name)


def _safe_log_loss(y_true: np.ndarray, probs: np.ndarray) -> float:
    probs = np.clip(probs, 1e-15, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return float(log_loss(y_true, probs, labels=LABELS))


def _draw_recall(y_true: np.ndarray, preds: np.ndarray) -> float:
    mask = y_true == 1
    return float((preds[mask] == 1).mean()) if mask.any() else 0.0


def _balanced_accuracy(cm: np.ndarray) -> float:
    recalls = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    return float(np.mean(recalls))


def _redistribute_knockout(probs: np.ndarray, stages: pd.Series | np.ndarray) -> np.ndarray:
    if isinstance(stages, pd.Series):
        is_knock = stages.isin(KNOCKOUT_STAGES).values
    else:
        is_knock = np.array([s in KNOCKOUT_STAGES for s in stages])
    out = probs.copy().astype(float)
    for i, knock in enumerate(is_knock):
        if knock:
            pw, _, pl = out[i]
            s = pw + pl
            out[i] = [pw / s, 0.0, pl / s] if s > 0 else [0.5, 0.0, 0.5]
    return out


def evaluate_probs(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    probs = np.clip(probs, 1e-15, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    cm = confusion_matrix(y_true, preds, labels=LABELS)
    return (
        _safe_log_loss(y_true, probs),
        float(accuracy_score(y_true, preds)),
        _draw_recall(y_true, preds),
        preds,
        cm,
    )


def confusion_text(cm: np.ndarray) -> str:
    rn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    lines = [
        "Counts:        Win  Draw  Loss",
        f"  Actual W    {cm[0,0]:>4}  {cm[0,1]:>4}  {cm[0,2]:>4}",
        f"  Actual D    {cm[1,0]:>4}  {cm[1,1]:>4}  {cm[1,2]:>4}",
        f"  Actual L    {cm[2,0]:>4}  {cm[2,1]:>4}  {cm[2,2]:>4}",
        "Recall:",
        f"  Actual W    {rn[0,0]:.2f}  {rn[0,1]:.2f}  {rn[0,2]:.2f}",
        f"  Actual D    {rn[1,0]:.2f}  {rn[1,1]:.2f}  {rn[1,2]:.2f}",
        f"  Actual L    {rn[2,0]:.2f}  {rn[2,1]:.2f}  {rn[2,2]:.2f}",
    ]
    return "\n".join(lines)


def load_raw_matches() -> list[dict]:
    if not RAW_MATCHES.exists():
        raise FileNotFoundError(
            f"Missing {RAW_MATCHES}. Run the old walk-forward pipeline once or restore the cache."
        )
    with open(RAW_MATCHES, "rb") as f:
        wc_train, wc_test, int_matches = pickle.load(f)
    rows: list[dict] = []
    for src, matches in [("world_cup", wc_train), ("world_cup", wc_test), ("international", int_matches)]:
        for m in matches:
            mm = dict(m)
            mm["home"] = norm_team(mm["home"])
            mm["away"] = norm_team(mm["away"])
            mm["date"] = pd.to_datetime(mm["date"])
            mm["year"] = int(mm["year"]) if mm.get("year") is not None else int(mm["date"].year)
            mm["competition"] = src
            mm["stage"] = mm.get("stage") if src == "world_cup" else "international"
            if "eff_winner" not in mm:
                if mm["score_home"] > mm["score_away"]:
                    mm["eff_winner"] = "home"
                elif mm["score_home"] < mm["score_away"]:
                    mm["eff_winner"] = "away"
                else:
                    mm["eff_winner"] = "draw"
            rows.append(mm)
    rows.sort(key=lambda x: x["date"])
    return rows


def supplemental_k(tournament: str) -> float:
    tournament = str(tournament)
    if tournament == "Friendly":
        return 20.0
    if "FIFA World Cup qualification" in tournament:
        return 50.0
    if "FIFA World Cup" in tournament:
        return 60.0
    if any(token in tournament for token in ["UEFA Euro", "Copa América", "African Cup", "Gold Cup", "Asian Cup", "Nations League"]):
        return 45.0
    return 35.0


def effective_winner_from_score(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "home"
    if away_score > home_score:
        return "away"
    return "draw"


def match_key(m: dict) -> tuple:
    return (
        pd.to_datetime(m["date"]).date().isoformat(),
        norm_team(m["home"]),
        norm_team(m["away"]),
        int(m["score_home"]),
        int(m["score_away"]),
    )


def load_supplemental_2026_matches(existing_matches: list[dict]) -> list[dict]:
    """Load post-cache matches used only for the final 2026 forecast."""
    if not SUPPLEMENTAL_2026_RESULTS.exists():
        print(f"      supplemental 2026 file missing: {SUPPLEMENTAL_2026_RESULTS}", flush=True)
        return []
    df = pd.read_csv(SUPPLEMENTAL_2026_RESULTS, parse_dates=["date"])
    df = df[(df["date"] >= pd.Timestamp("2022-01-01")) & (df["date"] < pd.Timestamp("2026-06-11"))].copy()
    df = df.dropna(subset=["home_score", "away_score", "home_team", "away_team"])
    existing = {match_key(m) for m in existing_matches}
    rows = []
    for _, r in df.sort_values("date").iterrows():
        home = norm_team(r["home_team"])
        away = norm_team(r["away_team"])
        sh = int(r["home_score"])
        sa = int(r["away_score"])
        m = {
            "home": home,
            "away": away,
            "date": pd.to_datetime(r["date"]),
            "year": int(pd.to_datetime(r["date"]).year),
            "score_home": sh,
            "score_away": sa,
            "competition": "international",
            "stage": "international",
            "tournament": str(r.get("tournament", "")),
            "country": str(r.get("country", "")),
            "neutral": bool(r.get("neutral", False)),
            "k": supplemental_k(str(r.get("tournament", ""))),
            "eff_winner": effective_winner_from_score(sh, sa),
            "source": "martj42_international_results_2022_2026",
        }
        key = match_key(m)
        if key in existing:
            continue
        rows.append(m)
        existing.add(key)
    rows.sort(key=lambda x: x["date"])
    return rows


def stage_to_fraction(stage: str) -> float:
    return {"group": 0.2, "R16": 0.4, "QF": 0.6, "SF": 0.8, "3rd": 0.8, "F": 1.0}.get(stage, 0.0)


def build_wc_history(matches: list[dict], max_year_exclusive: int = TEST_YEAR) -> dict[str, list[tuple[int, float]]]:
    deepest: dict[str, dict[int, float]] = defaultdict(dict)
    for m in matches:
        if m["competition"] != "world_cup" or m["year"] >= max_year_exclusive:
            continue
        frac = stage_to_fraction(m["stage"])
        for t in (m["home"], m["away"]):
            deepest[t][m["year"]] = max(deepest[t].get(m["year"], 0.0), frac)
    out: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for team, yd in deepest.items():
        out[team] = sorted(yd.items())
    return out


def wc_history_score(history: dict[str, list[tuple[int, float]]], team: str, current_year: int) -> float:
    vals = [(y, f) for y, f in history.get(team, []) if y < current_year]
    if not vals:
        return 0.0
    num = den = 0.0
    for y, f in vals:
        w = 0.85 ** ((current_year - y) / 4.0)
        num += w * f
        den += w
    return num / den if den else 0.0


class FeatureTracker:
    def __init__(self, initial_elo: float = 1500.0):
        self.ratings = defaultdict(lambda: initial_elo)
        self.recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        self.h2h: dict[frozenset, list[dict]] = defaultdict(list)
        self.upset_ema = defaultdict(float)
        self.giant_killer_ema = defaultdict(float)
        self.underdog_alpha = 0.15

    @staticmethod
    def expected(ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    @staticmethod
    def _stats(entries: list[dict], default_wr: float = 0.5) -> dict[str, float]:
        if not entries:
            return {"wr": default_wr, "gd": 0.0, "gs": 0.0, "gc": 0.0, "cs": 0.0, "opp_elo": 1500.0}
        n = len(entries)
        return {
            "wr": sum(e["win"] for e in entries) / n,
            "gd": sum(e["gs"] - e["gc"] for e in entries) / n,
            "gs": sum(e["gs"] for e in entries) / n,
            "gc": sum(e["gc"] for e in entries) / n,
            "cs": sum(e["cs"] for e in entries) / n,
            "opp_elo": sum(e["opp_elo"] for e in entries) / n,
        }

    def snapshot_features(
        self,
        home: str,
        away: str,
        date: pd.Timestamp,
        stage: str,
        year: int,
        history: dict[str, list[tuple[int, float]]],
        host_teams: set[str] | None = None,
    ) -> dict:
        home = norm_team(home)
        away = norm_team(away)
        rh, ra = self.ratings[home], self.ratings[away]
        if rh >= ra:
            a, b = home, away
        else:
            a, b = away, home
        elo_a, elo_b = self.ratings[a], self.ratings[b]
        cutoff_4y = date - pd.Timedelta(days=365 * 4)
        rec_a = list(self.recent[a])
        rec_b = list(self.recent[b])
        rec_a_4y = [e for e in rec_a if e["date"] >= cutoff_4y]
        rec_b_4y = [e for e in rec_b if e["date"] >= cutoff_4y]
        s5a, s5b = self._stats(rec_a[-5:]), self._stats(rec_b[-5:])
        s10a, s10b = self._stats(rec_a[-10:]), self._stats(rec_b[-10:])
        s4a, s4b = self._stats(rec_a_4y[-10:]), self._stats(rec_b_4y[-10:])

        h2h_entries = [e for e in self.h2h[frozenset((a, b))] if e["date"] >= cutoff_4y]
        wa = sum(1 for e in h2h_entries if e["winner"] == a)
        wb = sum(1 for e in h2h_entries if e["winner"] == b)
        dr = sum(1 for e in h2h_entries if e["winner"] is None)
        h2h_n = wa + wb + dr
        h2h_a = (wa + 0.5 * dr) / h2h_n if h2h_n else 0.5
        host_teams = host_teams or set()

        row = {
            "team_a": a,
            "team_b": b,
            "elo_a": elo_a,
            "elo_b": elo_b,
            "delta_elo": elo_a - elo_b,
            "elo_prob_a": self.expected(elo_a, elo_b),
            "wc_hist_a": wc_history_score(history, a, year),
            "wc_hist_b": wc_history_score(history, b, year),
            "h2h_4y_a": h2h_a,
            "h2h_4y_n": float(h2h_n),
            "stage_ord": STAGE_ORD.get(stage, 0),
            "is_group": 1 if stage == "group" else 0,
            "is_world_cup": 0 if stage == "international" else 1,
            "host_a": 1 if a in host_teams else 0,
            "host_b": 1 if b in host_teams else 0,
            "wr_a": s5a["wr"],
            "wr_b": s5b["wr"],
            "gd_a": s5a["gd"],
            "gd_b": s5b["gd"],
            "gs_a": s5a["gs"],
            "gs_b": s5b["gs"],
            "opp_elo_a": s5a["opp_elo"],
            "opp_elo_b": s5b["opp_elo"],
            "cs_a": s5a["cs"],
            "cs_b": s5b["cs"],
            "h2h_a": h2h_a,
            "upset_a": self.upset_ema[a],
            "gk_b": self.giant_killer_ema[b],
        }
        for prefix, sa, sb in [("5", s5a, s5b), ("10", s10a, s10b), ("4y", s4a, s4b)]:
            for key in ["wr", "gd", "gs", "gc", "cs", "opp_elo"]:
                row[f"{key}{prefix}_a"] = sa[key]
                row[f"{key}{prefix}_b"] = sb[key]
                row[f"{key}{prefix}_diff"] = sa[key] - sb[key]
        row["abs_delta_elo"] = abs(row["delta_elo"])
        row["delta_elo_sq"] = row["delta_elo"] ** 2
        row["abs_gd5_diff"] = abs(row["gd5_diff"])
        row["abs_wr5_diff"] = abs(row["wr5_diff"])
        row["abs_gs5_diff"] = abs(row["gs5_diff"])
        return row

    def update(self, home: str, away: str, sh: int, sa: int, k: float, eff_winner: str, date: pd.Timestamp) -> None:
        home = norm_team(home)
        away = norm_team(away)
        rh, ra = self.ratings[home], self.ratings[away]
        was_home_higher = rh >= ra
        higher = home if was_home_higher else away
        lower = away if was_home_higher else home
        if eff_winner is None:
            eff_winner = "home" if sh > sa else "away" if sh < sa else "draw"
        eh = self.expected(rh, ra)
        if sh > sa:
            score_h, score_a = 1.0, 0.0
        elif sh < sa:
            score_h, score_a = 0.0, 1.0
        else:
            score_h = score_a = 0.5
        self.ratings[home] = rh + k * (score_h - eh)
        self.ratings[away] = ra + k * (score_a - (1.0 - eh))
        self.recent[home].append({
            "date": date, "win": score_h, "gs": float(sh), "gc": float(sa),
            "cs": 1.0 if sa == 0 else 0.0, "opp_elo": ra,
        })
        self.recent[away].append({
            "date": date, "win": score_a, "gs": float(sa), "gc": float(sh),
            "cs": 1.0 if sh == 0 else 0.0, "opp_elo": rh,
        })
        winner = home if eff_winner == "home" else away if eff_winner == "away" else None
        self.h2h[frozenset((home, away))].append({"date": date, "winner": winner})
        if eff_winner != "draw":
            home_won = eff_winner == "home"
            higher_won = home_won == was_home_higher
            upset_event = 0.0 if higher_won else 1.0
            giant_killer_event = 1.0 if not higher_won else 0.0
            alpha = self.underdog_alpha
            self.upset_ema[higher] = (1 - alpha) * self.upset_ema[higher] + alpha * upset_event
            self.giant_killer_ema[lower] = (
                (1 - alpha) * self.giant_killer_ema[lower] + alpha * giant_killer_event
            )


def label_from_effective(feats: dict, home: str, eff_winner: str) -> int:
    if eff_winner == "draw":
        return 1
    a_side = "home" if feats["team_a"] == home else "away"
    return 0 if eff_winner == a_side else 2


def goals_from_a_perspective(feats: dict, home: str, sh: int, sa: int) -> tuple[int, int]:
    return (sh, sa) if feats["team_a"] == home else (sa, sh)


def build_feature_table(matches: list[dict], history_cutoff_year: int = TEST_YEAR) -> tuple[pd.DataFrame, FeatureTracker, dict]:
    wc_history = build_wc_history(matches, max_year_exclusive=history_cutoff_year)
    tracker = FeatureTracker()
    rows = []
    for m in matches:
        feats = tracker.snapshot_features(
            m["home"], m["away"], m["date"], m["stage"], int(m["year"]),
            wc_history, host_teams=set(),
        )
        ga, gb = goals_from_a_perspective(feats, m["home"], m["score_home"], m["score_away"])
        row = {
            **feats,
            "date": m["date"],
            "year": int(m["year"]),
            "stage": m["stage"],
            "competition": m["competition"],
            "home": m["home"],
            "away": m["away"],
            "score": f"{m['score_home']}-{m['score_away']}",
            "goals_a": ga,
            "goals_b": gb,
            "label": label_from_effective(feats, m["home"], m["eff_winner"]),
        }
        row["base_weight"] = 5.0 if m["competition"] == "world_cup" else 0.75
        rows.append(row)
        tracker.update(m["home"], m["away"], m["score_home"], m["score_away"], m["k"], m["eff_winner"], m["date"])
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df, tracker, wc_history


def fold_splits(df: pd.DataFrame, min_train_rows: int = MIN_TUNING_TRAIN_ROWS) -> list[tuple[pd.DataFrame, pd.DataFrame, int]]:
    folds = []
    world_cup_years = sorted(
        int(y) for y in df.loc[df["competition"] == "world_cup", "year"].dropna().unique()
        if int(y) < TEST_YEAR
    )
    for val_year in world_cup_years:
        val = df[(df["competition"] == "world_cup") & (df["year"] == val_year)].copy()
        first_val_date = val["date"].min()
        train = df[df["date"] < first_val_date].copy()
        if len(train) < min_train_rows:
            continue
        if train["label"].nunique() < len(LABELS):
            continue
        folds.append((train.reset_index(drop=True), val.reset_index(drop=True), val_year))
    return folds


def final_2022_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = df[(df["competition"] == "world_cup") & (df["year"] == TEST_YEAR)].copy()
    train = df[df["date"] < test["date"].min()].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def sample_weights(train: pd.DataFrame, target_year: int, half_life_years: float = 12.0) -> np.ndarray:
    age = np.maximum(0.0, float(target_year) - train["year"].astype(float).values)
    decay = 0.5 ** (age / half_life_years)
    return train["base_weight"].astype(float).values * decay


STRENGTH_FEATS = [
    "elo_a", "elo_b", "delta_elo", "elo_prob_a",
    "wc_hist_a", "wc_hist_b", "opp_elo5_a", "opp_elo5_b",
]
FORM5_FEATS = [
    "wr5_a", "wr5_b", "wr5_diff", "gd5_a", "gd5_b", "gd5_diff",
    "gs5_a", "gs5_b", "gc5_a", "gc5_b", "cs5_a", "cs5_b",
]
FORM10_FEATS = [
    "wr10_a", "wr10_b", "wr10_diff", "gd10_a", "gd10_b", "gd10_diff",
    "gs10_a", "gs10_b", "gc10_a", "gc10_b", "cs10_a", "cs10_b",
]
RECENT4Y_FEATS = [
    "wr4y_a", "wr4y_b", "wr4y_diff", "gd4y_a", "gd4y_b", "gd4y_diff",
    "gs4y_a", "gs4y_b", "gc4y_a", "gc4y_b", "cs4y_a", "cs4y_b",
]
CONTEXT_FEATS = [
    "h2h_4y_a", "h2h_4y_n", "stage_ord", "is_group", "is_world_cup",
    "abs_delta_elo", "delta_elo_sq", "abs_gd5_diff", "abs_wr5_diff",
    "upset_a", "gk_b",
]
CONTEXT_FEATS_NO_UNDERDOG = [f for f in CONTEXT_FEATS if f not in {"upset_a", "gk_b"}]
NO_ELO_FEATS = FORM5_FEATS + FORM10_FEATS + ["h2h_4y_a", "h2h_4y_n", "stage_ord", "is_group"]
FULL_FEATS = STRENGTH_FEATS + FORM5_FEATS + FORM10_FEATS + RECENT4Y_FEATS + CONTEXT_FEATS
FULL_FEATS_NO_UNDERDOG = (
    STRENGTH_FEATS + FORM5_FEATS + FORM10_FEATS + RECENT4Y_FEATS + CONTEXT_FEATS_NO_UNDERDOG
)
ORIGINAL_BASELINE_FEATS = ["delta_elo", "wr_a", "wr_b", "gd_a", "gd_b"]
ORIGINAL_DIRECT_18_FEATS = ORIGINAL_18_FEATURES.copy()
ORIGINAL_FORM_FEATS = ["wr_a", "wr_b", "gd_a", "gd_b", "gs_a", "gs_b", "cs_a", "cs_b"]
ORIGINAL_MATCHUP_FEATS = ["elo_a", "elo_b", "h2h_a", "upset_a", "gk_b", "stage_ord"]
FEATURE_BLOCKS_55 = {
    "strength_and_history": STRENGTH_FEATS,
    "last_5_form": FORM5_FEATS,
    "last_10_form": FORM10_FEATS,
    "recent_4y_form": RECENT4Y_FEATS,
    "context_and_interactions": CONTEXT_FEATS,
}

SCORE_FEATS = [
    "team_elo", "opp_elo", "elo_diff", "team_gs5", "team_gd5", "team_gc5",
    "team_cs5", "team_wc_hist", "opp_gd5", "opp_gc5", "opp_cs5",
    "is_group", "is_world_cup",
]


@dataclass
class ModelResult:
    name: str
    family: str
    features: list[str]
    best_config: dict
    fold_lls: list[float]
    fold_accs: list[float]
    fold_draw_recalls: list[float]
    test_ll: float
    test_acc: float
    test_draw_recall: float
    test_probs: np.ndarray
    test_preds: np.ndarray
    test_cm: np.ndarray
    fold_probs: list[np.ndarray] = field(default_factory=list)
    fold_y: list[np.ndarray] = field(default_factory=list)
    fitted: object | None = None
    scaler: object | None = None
    extra: dict = field(default_factory=dict)

    @property
    def mean_cv_ll(self) -> float:
        return float(np.mean(self.fold_lls))

    @property
    def std_cv_ll(self) -> float:
        return float(np.std(self.fold_lls))

    @property
    def mean_cv_acc(self) -> float:
        return float(np.mean(self.fold_accs))


def _scaled_arrays(train: pd.DataFrame, val: pd.DataFrame, feats: list[str]) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    xtr = scaler.fit_transform(train[feats].values)
    xva = scaler.transform(val[feats].values)
    return xtr, xva, scaler


def _full_proba(model, x: np.ndarray) -> np.ndarray:
    probs = model.predict_proba(x)
    if probs.shape[1] == 3:
        return probs
    out = np.zeros((len(x), 3))
    for j, c in enumerate(model.classes_):
        out[:, int(c)] = probs[:, j]
    return out


def fit_predict_mlr(train: pd.DataFrame, val: pd.DataFrame, feats: list[str], cfg: dict, target_year: int):
    xtr, xva, scaler = _scaled_arrays(train, val, feats)
    model = LogisticRegression(
        multi_class="multinomial", solver="lbfgs", max_iter=2500,
        C=cfg["C"], class_weight="balanced", random_state=RNG_SEED,
    )
    model.fit(xtr, train["label"].values, sample_weight=sample_weights(train, target_year))
    probs = _full_proba(model, xva)
    probs = _redistribute_knockout(probs, val["stage"])
    return probs, model, scaler


def fit_predict_rf(train: pd.DataFrame, val: pd.DataFrame, feats: list[str], cfg: dict, target_year: int):
    xtr = train[feats].values
    xva = val[feats].values
    model = RandomForestClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        min_samples_leaf=cfg["min_samples_leaf"],
        max_features=cfg.get("max_features", "sqrt"),
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RNG_SEED,
    )
    model.fit(xtr, train["label"].values, sample_weight=sample_weights(train, target_year))
    probs = _full_proba(model, xva)
    probs = _redistribute_knockout(probs, val["stage"])
    return probs, model, None


def to_goal_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        for side, opp, goals in [("a", "b", r["goals_a"]), ("b", "a", r["goals_b"])]:
            rows.append({
                "team_elo": r[f"elo_{side}"],
                "opp_elo": r[f"elo_{opp}"],
                "elo_diff": r[f"elo_{side}"] - r[f"elo_{opp}"],
                "team_gs5": r[f"gs5_{side}"],
                "team_gd5": r[f"gd5_{side}"],
                "team_gc5": r[f"gc5_{side}"],
                "team_cs5": r[f"cs5_{side}"],
                "team_wc_hist": r[f"wc_hist_{side}"],
                "opp_gd5": r[f"gd5_{opp}"],
                "opp_gc5": r[f"gc5_{opp}"],
                "opp_cs5": r[f"cs5_{opp}"],
                "is_group": r["is_group"],
                "is_world_cup": r["is_world_cup"],
                "goals": float(goals),
                "match_weight": r["fit_weight"],
            })
    return pd.DataFrame(rows)


def score_grid_from_lambdas(lam_a: float, lam_b: float, draw_boost: float = 0.0) -> np.ndarray:
    ia = poisson.pmf(np.arange(MAX_GOALS), max(lam_a, 1e-6))
    ib = poisson.pmf(np.arange(MAX_GOALS), max(lam_b, 1e-6))
    grid = np.outer(ia, ib)
    diag = np.eye(MAX_GOALS, dtype=bool)
    if draw_boost:
        grid[diag] *= math.exp(draw_boost)
    return grid / max(grid.sum(), 1e-15)


def wdl_from_lambdas(lam_a: float, lam_b: float, draw_boost: float = 0.0) -> tuple[float, float, float]:
    grid = score_grid_from_lambdas(lam_a, lam_b, draw_boost)
    i, j = np.indices(grid.shape)
    return float(grid[i > j].sum()), float(grid[i == j].sum()), float(grid[i < j].sum())


def fit_predict_poisson(train: pd.DataFrame, val: pd.DataFrame, feats: list[str], cfg: dict, target_year: int):
    train = train.copy()
    train["fit_weight"] = sample_weights(train, target_year)
    fit_rows = to_goal_rows(train)
    eval_rows = to_goal_rows(val.assign(fit_weight=1.0))
    xtr, xva, scaler = _scaled_arrays(fit_rows, eval_rows, feats)
    model = PoissonRegressor(alpha=cfg["alpha"], max_iter=1000)
    model.fit(xtr, fit_rows["goals"].values, sample_weight=fit_rows["match_weight"].values)
    lam = np.clip(model.predict(xva), 0.03, 7.0)
    probs = np.zeros((len(val), 3))
    lambdas = np.zeros((len(val), 2))
    for i in range(len(val)):
        la, lb = float(lam[2 * i]), float(lam[2 * i + 1])
        lambdas[i] = [la, lb]
        probs[i] = wdl_from_lambdas(la, lb, draw_boost=cfg.get("draw_boost", 0.0))
    probs = _redistribute_knockout(probs, val["stage"])
    return probs, model, scaler, lambdas


def fit_predict_nn(train: pd.DataFrame, val: pd.DataFrame, feats: list[str], cfg: dict, target_year: int):
    import torch
    import torch.nn as nn

    torch.set_num_threads(2)
    torch.manual_seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    random.seed(RNG_SEED)

    xtr, xva, scaler = _scaled_arrays(train, val, feats)
    ytr = train["label"].values.astype(np.int64)
    yva = val["label"].values.astype(np.int64)
    w = sample_weights(train, target_year).astype(np.float32)
    w = w / max(float(w.mean()), 1e-8)
    class_counts = np.bincount(ytr, minlength=3)
    class_w = len(ytr) / (3 * np.maximum(class_counts, 1))
    per_sample_class_w = class_w[ytr].astype(np.float32)
    w = w * per_sample_class_w

    xtr_t = torch.tensor(xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    w_t = torch.tensor(w, dtype=torch.float32)
    xva_t = torch.tensor(xva, dtype=torch.float32)

    model = nn.Sequential(
        nn.Linear(len(feats), cfg["hidden1"]),
        nn.ReLU(),
        nn.Dropout(cfg["dropout"]),
        nn.Linear(cfg["hidden1"], cfg["hidden2"]),
        nn.ReLU(),
        nn.Dropout(cfg["dropout"]),
        nn.Linear(cfg["hidden2"], 3),
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    best_ll = float("inf")
    best_state = None
    patience = cfg.get("patience", 20)
    stale = 0
    batch = min(256, len(xtr))
    for _epoch in range(cfg.get("epochs", 220)):
        idx = torch.randperm(len(xtr_t))
        model.train()
        for start in range(0, len(idx), batch):
            sl = idx[start:start + batch]
            opt.zero_grad()
            loss = (loss_fn(model(xtr_t[sl]), ytr_t[sl]) * w_t[sl]).mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(xva_t), dim=1).numpy()
        probs = _redistribute_knockout(probs, val["stage"])
        ll = _safe_log_loss(yva, probs)
        if ll < best_ll - 1e-5:
            best_ll = ll
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(xva_t), dim=1).numpy()
    probs = _redistribute_knockout(probs, val["stage"])
    return probs, model, scaler


def evaluate_spec(
    df: pd.DataFrame,
    name: str,
    family: str,
    feats: list[str],
    grid: list[dict],
    fit_fn: Callable,
) -> ModelResult:
    print(f"[model] {name}: {len(grid)} configs, {len(feats)} features", flush=True)
    folds = fold_splits(df)
    best_cfg = None
    best_mean = float("inf")
    for cfg in grid:
        y_all = []
        p_all = []
        for train, val, val_year in folds:
            out = fit_fn(train, val, feats, cfg, val_year)
            probs = out[0]
            y_all.append(val["label"].values)
            p_all.append(probs)
        mean_ll = _safe_log_loss(np.concatenate(y_all), np.vstack(p_all))
        if mean_ll < best_mean:
            best_mean = mean_ll
            best_cfg = cfg
    print(f"[model] {name}: best={best_cfg} mean_cv_ll={best_mean:.4f}", flush=True)

    fold_lls, fold_accs, fold_draws, fold_probs, fold_y = [], [], [], [], []
    for train, val, val_year in folds:
        out = fit_fn(train, val, feats, best_cfg, val_year)
        probs = out[0]
        ll, acc, drec, _, _ = evaluate_probs(val["label"].values, probs)
        fold_lls.append(ll)
        fold_accs.append(acc)
        fold_draws.append(drec)
        fold_probs.append(probs)
        fold_y.append(val["label"].values)

    train_final, test = final_2022_split(df)
    out = fit_fn(train_final, test, feats, best_cfg, TEST_YEAR)
    test_probs = out[0]
    fitted = out[1] if len(out) > 1 else None
    scaler = out[2] if len(out) > 2 else None
    ll, acc, drec, preds, cm = evaluate_probs(test["label"].values, test_probs)
    return ModelResult(
        name=name, family=family, features=feats, best_config=best_cfg,
        fold_lls=fold_lls, fold_accs=fold_accs, fold_draw_recalls=fold_draws,
        test_ll=ll, test_acc=acc, test_draw_recall=drec, test_probs=test_probs,
        test_preds=preds, test_cm=cm, fold_probs=fold_probs, fold_y=fold_y,
        fitted=fitted, scaler=scaler,
    )


def tune_ensemble(models: list[ModelResult], test_y: np.ndarray) -> ModelResult:
    candidates = []
    for fam in ["Poisson", "MLR", "RF", "NN"]:
        fam_models = [m for m in models if m.family == fam]
        if fam_models:
            candidates.append(min(fam_models, key=lambda m: m.mean_cv_ll))
    candidates = sorted(candidates, key=lambda m: m.mean_cv_ll)[:4]
    folds_n = len(candidates[0].fold_probs)
    y_folds = candidates[0].fold_y
    grid = np.arange(0.0, 1.0 + 1e-9, 0.1)
    best_w = None
    best_ll = float("inf")
    if len(candidates) == 1:
        best_w = [1.0]
    else:
        for raw in product(grid, repeat=len(candidates)):
            if abs(sum(raw) - 1.0) > 1e-9:
                continue
            lls = []
            for f in range(folds_n):
                p = sum(w * m.fold_probs[f] for w, m in zip(raw, candidates))
                lls.append(_safe_log_loss(y_folds[f], p))
            mean_ll = float(np.mean(lls))
            if mean_ll < best_ll:
                best_ll = mean_ll
                best_w = list(map(float, raw))
    ensemble_fold_probs, ensemble_fold_y = [], []
    ensemble_fold_lls, ensemble_fold_accs, ensemble_fold_draws = [], [], []
    for f in range(folds_n):
        p = sum(w * m.fold_probs[f] for w, m in zip(best_w, candidates))
        ll, acc, drec, _, _ = evaluate_probs(y_folds[f], p)
        ensemble_fold_probs.append(p)
        ensemble_fold_y.append(y_folds[f])
        ensemble_fold_lls.append(ll)
        ensemble_fold_accs.append(acc)
        ensemble_fold_draws.append(drec)

    test_probs = sum(w * m.test_probs for w, m in zip(best_w, candidates))
    ll, acc, drec, preds, cm = evaluate_probs(test_y, test_probs)
    return ModelResult(
        name="Validation-tuned family ensemble",
        family="Ensemble",
        features=[],
        best_config={"members": [m.name for m in candidates], "weights": best_w},
        fold_lls=ensemble_fold_lls,
        fold_accs=ensemble_fold_accs,
        fold_draw_recalls=ensemble_fold_draws,
        test_ll=ll,
        test_acc=acc,
        test_draw_recall=drec,
        test_probs=test_probs,
        test_preds=preds,
        test_cm=cm,
        fold_probs=ensemble_fold_probs,
        fold_y=ensemble_fold_y,
    )


ORIGINAL_FEATURE_MAP = {
    "elo_a": {"elo_a", "delta_elo", "elo_prob_a", "abs_delta_elo", "delta_elo_sq", "team_elo", "elo_diff"},
    "elo_b": {"elo_b", "delta_elo", "elo_prob_a", "abs_delta_elo", "delta_elo_sq", "opp_elo", "elo_diff"},
    "wr_a": {"wr_a", "wr5_a", "wr10_a", "wr4y_a", "wr5_diff", "wr10_diff", "wr4y_diff", "abs_wr5_diff"},
    "wr_b": {"wr_b", "wr5_b", "wr10_b", "wr4y_b", "wr5_diff", "wr10_diff", "wr4y_diff", "abs_wr5_diff"},
    "gd_a": {"gd_a", "gd5_a", "gd10_a", "gd4y_a", "gd5_diff", "gd10_diff", "gd4y_diff", "abs_gd5_diff", "team_gd5"},
    "gd_b": {"gd_b", "gd5_b", "gd10_b", "gd4y_b", "gd5_diff", "gd10_diff", "gd4y_diff", "abs_gd5_diff", "team_gd5", "opp_gd5"},
    "gs_a": {"gs_a", "gs5_a", "gs10_a", "gs4y_a", "gs5_diff", "team_gs5"},
    "gs_b": {"gs_b", "gs5_b", "gs10_b", "gs4y_b", "gs5_diff", "team_gs5"},
    "opp_elo_a": {"opp_elo_a", "opp_elo5_a", "opp_elo10_a", "opp_elo4y_a", "opp_elo5_diff"},
    "opp_elo_b": {"opp_elo_b", "opp_elo5_b", "opp_elo10_b", "opp_elo4y_b", "opp_elo5_diff"},
    "cs_a": {"cs_a", "cs5_a", "cs10_a", "cs4y_a", "cs5_diff", "team_cs5"},
    "cs_b": {"cs_b", "cs5_b", "cs10_b", "cs4y_b", "cs5_diff", "team_cs5", "opp_cs5"},
    "wc_hist_a": {"wc_hist_a", "team_wc_hist"},
    "wc_hist_b": {"wc_hist_b", "team_wc_hist"},
    "h2h_a": {"h2h_a", "h2h_4y_a", "h2h_4y_n"},
    "stage_ord": {"stage_ord", "is_group", "is_world_cup"},
    "upset_a": {"upset_a"},
    "gk_b": {"gk_b"},
}

DERIVED_SOURCE_FAMILIES = {
    "delta_elo": ["elo_a", "elo_b"],
    "elo_prob_a": ["elo_a", "elo_b"],
    "abs_delta_elo": ["elo_a", "elo_b"],
    "delta_elo_sq": ["elo_a", "elo_b"],
    "wr5_diff": ["wr_a", "wr_b"],
    "wr10_diff": ["wr_a", "wr_b"],
    "wr4y_diff": ["wr_a", "wr_b"],
    "abs_wr5_diff": ["wr_a", "wr_b"],
    "gd5_diff": ["gd_a", "gd_b"],
    "gd10_diff": ["gd_a", "gd_b"],
    "gd4y_diff": ["gd_a", "gd_b"],
    "abs_gd5_diff": ["gd_a", "gd_b"],
    "gs5_diff": ["gs_a", "gs_b"],
    "opp_elo5_diff": ["opp_elo_a", "opp_elo_b"],
    "cs5_diff": ["cs_a", "cs_b"],
    "is_group": ["stage_ord"],
    "is_world_cup": ["stage_ord"],
}


def original_features_used(feats: list[str]) -> list[str]:
    fset = set(feats)
    used = []
    for original in ORIGINAL_18_FEATURES:
        if fset.intersection(ORIGINAL_FEATURE_MAP[original]):
            used.append(original)
    return used


def comparison_table(models: list[ModelResult]) -> pd.DataFrame:
    lookup = {m.name: m for m in models}
    rows = []
    for m in models:
        feats = features_for_usage(m, lookup)
        used_original = original_features_used(feats)
        rows.append({
            "Model": m.name,
            "Family": m.family,
            "#Feat": len(feats),
            "Original18 Used": len(used_original),
            "Best Config": json.dumps(m.best_config),
            "Mean CV LL": round(m.mean_cv_ll, 4),
            "Std CV LL": round(m.std_cv_ll, 4),
            "Mean CV Acc": round(m.mean_cv_acc, 4) if not math.isnan(m.mean_cv_acc) else np.nan,
            "Test LL": round(m.test_ll, 4),
            "Test Acc": round(m.test_acc, 4),
            "Draw Recall": round(m.test_draw_recall, 4),
            "CV-Test Gap": round(m.test_ll - m.mean_cv_ll, 4),
        })
    return pd.DataFrame(rows).sort_values(["Test LL", "Test Acc"], ascending=[True, False]).reset_index(drop=True)


def features_for_usage(model: ModelResult, lookup: dict[str, ModelResult]) -> list[str]:
    if model.family != "Ensemble":
        return model.features
    members = model.best_config.get("members", [])
    union: set[str] = set()
    for member in members:
        if member in lookup:
            union.update(lookup[member].features)
    return sorted(union)


def write_feature_artifacts(models: list[ModelResult]) -> pd.DataFrame:
    glossary_rows = []
    for feat in ORIGINAL_18_FEATURES:
        family, definition = FEATURE_GLOSSARY[feat]
        glossary_rows.append({
            "Feature": feat,
            "Feature Family": family,
            "Definition": definition,
            "Final Pipeline Representation": ", ".join(sorted(ORIGINAL_FEATURE_MAP[feat])),
        })
    pd.DataFrame(glossary_rows).to_csv(OUT / "feature_glossary_18.csv", index=False)

    block_rows = []
    for block, columns in FEATURE_BLOCKS_55.items():
        for col in columns:
            source_families = [
                feat for feat in ORIGINAL_18_FEATURES
                if col in ORIGINAL_FEATURE_MAP[feat]
            ]
            source_families.extend(DERIVED_SOURCE_FAMILIES.get(col, []))
            source_families = sorted(set(source_families), key=ORIGINAL_18_FEATURES.index)
            block_rows.append({
                "Feature Block": block,
                "Direct Column": col,
                "Source 18-Family Interpretation": ", ".join(source_families) if source_families else "derived context",
            })
    pd.DataFrame(block_rows).to_csv(OUT / "feature_blocks_55.csv", index=False)

    lookup = {m.name: m for m in models}
    usage_rows = []
    for m in models:
        feats = features_for_usage(m, lookup)
        used = set(original_features_used(feats))
        row = {
            "Model": m.name,
            "Family": m.family,
            "Direct Feature Count": len(feats),
            "Original18 Used Count": len(used),
            "Original18 Used": ", ".join([f for f in ORIGINAL_18_FEATURES if f in used]),
            "Original18 Not Used": ", ".join([f for f in ORIGINAL_18_FEATURES if f not in used]),
        }
        for feat in ORIGINAL_18_FEATURES:
            row[feat] = 1 if feat in used else 0
        usage_rows.append(row)
    usage = pd.DataFrame(usage_rows)
    usage.to_csv(OUT / "feature_usage.csv", index=False)
    usage[["Model", "Family", "Direct Feature Count", "Original18 Used Count"] + ORIGINAL_18_FEATURES].to_csv(
        OUT / "feature_usage_matrix.csv", index=False
    )
    return usage


def write_fold_metrics(models: list[ModelResult], df: pd.DataFrame) -> pd.DataFrame:
    lookup = {m.name: m for m in models}
    rows = []
    folds = fold_splits(df)
    for m in models:
        feats = features_for_usage(m, lookup)
        used_count = len(original_features_used(feats))
        if len(m.fold_lls) == len(folds):
            for i, (train, val, val_year) in enumerate(folds):
                rows.append({
                    "Model": m.name,
                    "Family": m.family,
                    "Evaluation": "CV",
                    "Eval World Cup": val_year,
                    "Train Start": train["date"].min().date().isoformat(),
                    "Train End": train["date"].max().date().isoformat(),
                    "Eval Start": val["date"].min().date().isoformat(),
                    "Eval End": val["date"].max().date().isoformat(),
                    "Training Rows": len(train),
                    "World Cup Training Rows": int((train["competition"] == "world_cup").sum()),
                    "International Training Rows": int((train["competition"] == "international").sum()),
                    "Evaluation Matches": len(val),
                    "Direct Feature Count": len(feats),
                    "Original18 Used Count": used_count,
                    "Log Loss": round(float(m.fold_lls[i]), 4),
                    "Accuracy": round(float(m.fold_accs[i]), 4),
                    "Draw Recall": round(float(m.fold_draw_recalls[i]), 4),
                })

        train, test = final_2022_split(df)
        rows.append({
            "Model": m.name,
            "Family": m.family,
            "Evaluation": "Final Test",
            "Eval World Cup": TEST_YEAR,
            "Train Start": train["date"].min().date().isoformat(),
            "Train End": train["date"].max().date().isoformat(),
            "Eval Start": test["date"].min().date().isoformat(),
            "Eval End": test["date"].max().date().isoformat(),
            "Training Rows": len(train),
            "World Cup Training Rows": int((train["competition"] == "world_cup").sum()),
            "International Training Rows": int((train["competition"] == "international").sum()),
            "Evaluation Matches": len(test),
            "Direct Feature Count": len(feats),
            "Original18 Used Count": used_count,
            "Log Loss": round(float(m.test_ll), 4),
            "Accuracy": round(float(m.test_acc), 4),
            "Draw Recall": round(float(m.test_draw_recall), 4),
        })
    fold_metrics = pd.DataFrame(rows)
    fold_metrics.to_csv(OUT / "fold_metrics.csv", index=False)
    return fold_metrics


def write_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    train_2022, test_2022 = final_2022_split(df)
    rows = [
        {
            "Split": "All supervised rows",
            "Rows": len(df),
            "World Cup Rows": int((df["competition"] == "world_cup").sum()),
            "International Rows": int((df["competition"] == "international").sum()),
            "Start": df["date"].min().date().isoformat(),
            "End": df["date"].max().date().isoformat(),
        },
        {
            "Split": "Pre-2022 training pool",
            "Rows": len(train_2022),
            "World Cup Rows": int((train_2022["competition"] == "world_cup").sum()),
            "International Rows": int((train_2022["competition"] == "international").sum()),
            "Start": train_2022["date"].min().date().isoformat(),
            "End": train_2022["date"].max().date().isoformat(),
        },
        {
            "Split": "Held-out 2022 World Cup",
            "Rows": len(test_2022),
            "World Cup Rows": int((test_2022["competition"] == "world_cup").sum()),
            "International Rows": int((test_2022["competition"] == "international").sum()),
            "Start": test_2022["date"].min().date().isoformat(),
            "End": test_2022["date"].max().date().isoformat(),
        },
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "dataset_summary.csv", index=False)
    return summary


def write_2026_training_summary(base_df: pd.DataFrame, augmented_df: pd.DataFrame, supplemental: list[dict]) -> pd.DataFrame:
    rows = [
        {
            "Split": "Base model-selection table",
            "Rows": len(base_df),
            "World Cup Rows": int((base_df["competition"] == "world_cup").sum()),
            "International Rows": int((base_df["competition"] == "international").sum()),
            "Start": base_df["date"].min().date().isoformat(),
            "End": base_df["date"].max().date().isoformat(),
        },
        {
            "Split": "2026-only supplemental rows",
            "Rows": len(supplemental),
            "World Cup Rows": 0,
            "International Rows": len(supplemental),
            "Start": min(m["date"] for m in supplemental).date().isoformat() if supplemental else "",
            "End": max(m["date"] for m in supplemental).date().isoformat() if supplemental else "",
        },
        {
            "Split": "2026 augmented training table",
            "Rows": len(augmented_df),
            "World Cup Rows": int((augmented_df["competition"] == "world_cup").sum()),
            "International Rows": int((augmented_df["competition"] == "international").sum()),
            "Start": augmented_df["date"].min().date().isoformat(),
            "End": augmented_df["date"].max().date().isoformat(),
        },
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "training_2026_augmented_summary.csv", index=False)
    return summary


def label_counts(df: pd.DataFrame) -> pd.Series:
    return df["label"].map(LABEL_STR).value_counts().reindex(["Win", "Draw", "Loss"]).fillna(0)


def write_figures(
    comp: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    feature_usage: pd.DataFrame,
    champ: pd.DataFrame,
    group_adv: pd.DataFrame,
    df: pd.DataFrame,
) -> None:
    mpl_config = FIG_DIR / "_mplconfig"
    mpl_cache = FIG_DIR / "_cache"
    mpl_config.mkdir(parents=True, exist_ok=True)
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    os.environ.setdefault("XDG_CACHE_HOME", str(mpl_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 220,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
    })

    train_2022, test_2022 = final_2022_split(df)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    comp_counts = df["competition"].value_counts().reindex(["international", "world_cup"]).fillna(0)
    axes[0, 0].bar(["International", "World Cup"], comp_counts.values, color=["#52796f", "#cad2c5"])
    axes[0, 0].set_title("Supervised Rows by Competition")
    axes[0, 0].set_ylabel("Matches")

    wc_counts = df[df["competition"] == "world_cup"].groupby("year").size()
    axes[0, 1].bar(wc_counts.index.astype(str), wc_counts.values, color="#354f52")
    axes[0, 1].set_title("World Cup Match Rows by Tournament")
    axes[0, 1].tick_params(axis="x", rotation=75)

    tr_counts = label_counts(train_2022)
    axes[1, 0].bar(tr_counts.index, tr_counts.values, color=["#2f3e46", "#84a98c", "#cad2c5"])
    axes[1, 0].set_title("Pre-2022 Label Distribution")
    axes[1, 0].set_ylabel("Matches")

    te_counts = label_counts(test_2022)
    axes[1, 1].bar(te_counts.index, te_counts.values, color=["#2f3e46", "#84a98c", "#cad2c5"])
    axes[1, 1].set_title("Held-out 2022 Label Distribution")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dataset_overview.png", bbox_inches="tight")
    plt.close(fig)

    ordered = comp.sort_values("Test LL", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    axes[0].barh(ordered["Model"], ordered["Test LL"], color="#52796f")
    axes[0].set_xlabel("2022 log loss")
    axes[0].set_title("Lower Is Better")
    axes[1].barh(ordered["Model"], ordered["Test Acc"], color="#2f3e46")
    axes[1].set_xlabel("2022 accuracy")
    axes[1].set_title("Higher Is Better")
    axes[1].set_xlim(0.0, 0.7)
    fig.suptitle("Held-out 2022 Model Performance")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_test_metrics.png", bbox_inches="tight")
    plt.close(fig)

    selected = [
        "MLR-full",
        "RF-full",
        "NN-strength",
        "NN-full-regularized",
        "Poisson-independent",
        "Validation-tuned family ensemble",
    ]
    cv = fold_metrics[fold_metrics["Evaluation"] == "CV"].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    for model in selected:
        sub = cv[cv["Model"] == model].sort_values("Eval World Cup")
        if not sub.empty:
            ax.plot(sub["Eval World Cup"], sub["Log Loss"], marker="o", label=model)
    ax.set_title("Train Before Tournament, Test on Next World Cup")
    ax.set_xlabel("Validation World Cup")
    ax.set_ylabel("Log loss")
    years = sorted(cv["Eval World Cup"].unique())
    ax.set_xticks(years[::2] if len(years) > 10 else years)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fold_log_loss.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for model in selected:
        sub = cv[cv["Model"] == model].sort_values("Eval World Cup")
        if not sub.empty:
            ax.plot(sub["Eval World Cup"], sub["Accuracy"], marker="o", label=model)
    ax.set_title("Walk-forward Accuracy by Next World Cup")
    ax.set_xlabel("Validation World Cup")
    ax.set_ylabel("Accuracy")
    years = sorted(cv["Eval World Cup"].unique())
    ax.set_xticks(years[::2] if len(years) > 10 else years)
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0.35, 0.75)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fold_accuracy.png", bbox_inches="tight")
    plt.close(fig)

    matrix = feature_usage[ORIGINAL_18_FEATURES].values.astype(float)
    ylabels = [
        f"{row.Model} ({int(row['Original18 Used Count'])}/18)"
        for _, row in feature_usage.iterrows()
    ]
    fig, ax = plt.subplots(figsize=(12, max(5, 0.42 * len(feature_usage))))
    ax.imshow(matrix, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(ORIGINAL_18_FEATURES)))
    ax.set_xticklabels(ORIGINAL_18_FEATURES, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_title("Original 18 Engineered Feature Families Used by Each Model")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, "1" if matrix[i, j] else "", ha="center", va="center", fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_usage_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    top_champ = champ.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_champ["team"], top_champ["champion_probability"], color="#52796f")
    ax.set_xlabel("Champion probability")
    ax.set_title("2026 Monte Carlo Champion Probabilities")
    ax.set_xlim(0, max(0.28, float(top_champ["champion_probability"].max()) * 1.1))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "champion_probabilities_2026.png", bbox_inches="tight")
    plt.close(fig)

    groups = sorted(group_adv["group"].unique())
    fig, axes = plt.subplots(3, 4, figsize=(14, 8), sharex=True)
    for ax, group in zip(axes.ravel(), groups):
        sub = group_adv[group_adv["group"] == group].sort_values("p_advance")
        ax.barh(sub["team"], sub["p_advance"], color="#84a98c")
        ax.set_title(f"Group {group}")
        ax.set_xlim(0, 1)
        ax.tick_params(axis="y", labelsize=7)
    for ax in axes.ravel()[len(groups):]:
        ax.axis("off")
    fig.suptitle("2026 Group Advancement Probabilities")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "group_advancement_2026.png", bbox_inches="tight")
    plt.close(fig)


def fit_final_2026_score_model(df: pd.DataFrame, score_result: ModelResult):
    train = df[df["date"] <= df["date"].max()].copy()
    cfg = score_result.best_config
    train["fit_weight"] = sample_weights(train, 2026)
    fit_rows = to_goal_rows(train)
    scaler = StandardScaler()
    x = scaler.fit_transform(fit_rows[SCORE_FEATS].values)
    model = PoissonRegressor(alpha=cfg["alpha"], max_iter=1000)
    model.fit(x, fit_rows["goals"].values, sample_weight=fit_rows["match_weight"].values)
    return model, scaler, cfg


def _new_nn_model(n_features: int, cfg: dict):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(n_features, cfg["hidden1"]),
        nn.ReLU(),
        nn.Dropout(cfg["dropout"]),
        nn.Linear(cfg["hidden1"], cfg["hidden2"]),
        nn.ReLU(),
        nn.Dropout(cfg["dropout"]),
        nn.Linear(cfg["hidden2"], 3),
    )


def _nn_sample_weights(train: pd.DataFrame, target_year: int) -> np.ndarray:
    ytr = train["label"].values.astype(np.int64)
    w = sample_weights(train, target_year).astype(np.float32)
    w = w / max(float(w.mean()), 1e-8)
    class_counts = np.bincount(ytr, minlength=3)
    class_w = len(ytr) / (3 * np.maximum(class_counts, 1))
    return w * class_w[ytr].astype(np.float32)


def fit_final_2026_nn(df: pd.DataFrame, feats: list[str], cfg: dict):
    import torch
    import torch.nn as nn

    torch.set_num_threads(2)
    torch.manual_seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    random.seed(RNG_SEED)

    data = df.sort_values("date").reset_index(drop=True)
    val_n = min(max(256, int(0.12 * len(data))), max(1, len(data) // 4))
    tune_train = data.iloc[:-val_n].copy()
    tune_val = data.iloc[-val_n:].copy()

    scaler_tune = StandardScaler()
    xtr = scaler_tune.fit_transform(tune_train[feats].values)
    xva = scaler_tune.transform(tune_val[feats].values)
    ytr = tune_train["label"].values.astype(np.int64)
    yva = tune_val["label"].values.astype(np.int64)
    w = _nn_sample_weights(tune_train, 2026)

    xtr_t = torch.tensor(xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    w_t = torch.tensor(w, dtype=torch.float32)
    xva_t = torch.tensor(xva, dtype=torch.float32)

    model = _new_nn_model(len(feats), cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    patience = cfg.get("patience", 20)
    batch = min(256, len(xtr_t))
    best_ll = float("inf")
    best_epoch = 1
    stale = 0
    for epoch in range(cfg.get("epochs", 220)):
        idx = torch.randperm(len(xtr_t))
        model.train()
        for start in range(0, len(idx), batch):
            sl = idx[start:start + batch]
            opt.zero_grad()
            loss = (loss_fn(model(xtr_t[sl]), ytr_t[sl]) * w_t[sl]).mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(xva_t), dim=1).numpy()
        probs = _redistribute_knockout(probs, tune_val["stage"])
        ll = _safe_log_loss(yva, probs)
        if ll < best_ll - 1e-5:
            best_ll = ll
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    torch.manual_seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    random.seed(RNG_SEED)
    scaler = StandardScaler()
    xall = scaler.fit_transform(data[feats].values)
    yall = data["label"].values.astype(np.int64)
    wall = _nn_sample_weights(data, 2026)
    xall_t = torch.tensor(xall, dtype=torch.float32)
    yall_t = torch.tensor(yall, dtype=torch.long)
    wall_t = torch.tensor(wall, dtype=torch.float32)
    model = _new_nn_model(len(feats), cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    batch = min(256, len(xall_t))
    for _epoch in range(best_epoch):
        idx = torch.randperm(len(xall_t))
        model.train()
        for start in range(0, len(idx), batch):
            sl = idx[start:start + batch]
            opt.zero_grad()
            loss = (loss_fn(model(xall_t[sl]), yall_t[sl]) * wall_t[sl]).mean()
            loss.backward()
            opt.step()
    model.eval()
    return model, scaler, {**cfg, "final_epochs": best_epoch, "internal_validation_ll": float(best_ll)}


def fit_final_2026_direct_model(df: pd.DataFrame, result: ModelResult):
    train = df.sort_values("date").copy()
    feats = result.features
    cfg = result.best_config
    if result.family == "MLR":
        scaler = StandardScaler()
        x = scaler.fit_transform(train[feats].values)
        model = LogisticRegression(
            multi_class="multinomial", solver="lbfgs", max_iter=2500,
            C=cfg["C"], class_weight="balanced", random_state=RNG_SEED,
        )
        model.fit(x, train["label"].values, sample_weight=sample_weights(train, 2026))
        return model, scaler, cfg
    if result.family == "RF":
        model = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_leaf=cfg["min_samples_leaf"],
            max_features=cfg.get("max_features", "sqrt"),
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RNG_SEED,
        )
        model.fit(train[feats].values, train["label"].values, sample_weight=sample_weights(train, 2026))
        return model, None, cfg
    if result.family == "NN":
        return fit_final_2026_nn(train, feats, cfg)
    raise ValueError(f"Unsupported direct forecast model family: {result.family}")


def fit_final_2026_outcome_predictor(
    df: pd.DataFrame,
    result: ModelResult,
    results_by_name: dict[str, ModelResult],
    cache: dict[str, dict],
) -> dict:
    if result.name in cache:
        return cache[result.name]
    if result.family == "Ensemble":
        members = [
            fit_final_2026_outcome_predictor(df, results_by_name[name], results_by_name, cache)
            for name in result.best_config["members"]
        ]
        predictor = {
            "kind": "ensemble",
            "name": result.name,
            "members": members,
            "weights": result.best_config["weights"],
        }
    elif result.family == "Poisson":
        model, scaler, cfg = fit_final_2026_score_model(df, result)
        predictor = {
            "kind": "poisson",
            "name": result.name,
            "features": result.features,
            "model": model,
            "scaler": scaler,
            "cfg": cfg,
        }
    else:
        model, scaler, cfg = fit_final_2026_direct_model(df, result)
        predictor = {
            "kind": result.family.lower(),
            "name": result.name,
            "features": result.features,
            "model": model,
            "scaler": scaler,
            "cfg": cfg,
        }
    cache[result.name] = predictor
    return predictor


def tracker_after_matches(matches: list[dict], history_cutoff_year: int = TEST_YEAR) -> tuple[FeatureTracker, dict]:
    history = build_wc_history(matches, max_year_exclusive=history_cutoff_year)
    tracker = FeatureTracker()
    for m in sorted(matches, key=lambda x: x["date"]):
        tracker.update(m["home"], m["away"], m["score_home"], m["score_away"], m["k"], m["eff_winner"], m["date"])
    return tracker, history


def score_features_for_match(row: dict) -> pd.DataFrame:
    rows = []
    for side, opp in [("a", "b"), ("b", "a")]:
        rows.append({
            "team_elo": row[f"elo_{side}"],
            "opp_elo": row[f"elo_{opp}"],
            "elo_diff": row[f"elo_{side}"] - row[f"elo_{opp}"],
            "team_gs5": row[f"gs5_{side}"],
            "team_gd5": row[f"gd5_{side}"],
            "team_gc5": row[f"gc5_{side}"],
            "team_cs5": row[f"cs5_{side}"],
            "team_wc_hist": row[f"wc_hist_{side}"],
            "opp_gd5": row[f"gd5_{opp}"],
            "opp_gc5": row[f"gc5_{opp}"],
            "opp_cs5": row[f"cs5_{opp}"],
            "is_group": row["is_group"],
            "is_world_cup": row["is_world_cup"],
        })
    return pd.DataFrame(rows)


def predict_score_probs(model, scaler, cfg: dict, tracker: FeatureTracker, history: dict, team1: str, team2: str, stage: str) -> tuple[np.ndarray, tuple[float, float], dict]:
    team1 = norm_team(team1)
    team2 = norm_team(team2)
    feats = tracker.snapshot_features(
        team1, team2, pd.Timestamp("2026-06-11"), stage, 2026,
        history, host_teams={norm_team(t) for t in HOSTS_2026},
    )
    x = scaler.transform(score_features_for_match(feats)[SCORE_FEATS].values)
    lam = np.clip(model.predict(x), 0.03, 7.0)
    probs_a = np.array(wdl_from_lambdas(float(lam[0]), float(lam[1]), cfg.get("draw_boost", 0.0)))
    if stage != "group":
        probs_a = _redistribute_knockout(probs_a.reshape(1, -1), np.array([stage]))[0]
    if feats["team_a"] == team1:
        probs = probs_a
        lambdas = (float(lam[0]), float(lam[1]))
    else:
        # Convert from team_a/team_b perspective to the requested team1/team2 order.
        probs = np.array([probs_a[2], probs_a[1], probs_a[0]])
        lambdas = (float(lam[1]), float(lam[0]))
    return probs, lambdas, feats


def predict_outcome_probs(predictor: dict, tracker: FeatureTracker, history: dict, team1: str, team2: str, stage: str) -> np.ndarray:
    team1 = norm_team(team1)
    team2 = norm_team(team2)
    if predictor["kind"] == "ensemble":
        probs = sum(
            float(w) * predict_outcome_probs(member, tracker, history, team1, team2, stage)
            for w, member in zip(predictor["weights"], predictor["members"])
        )
        return probs / max(float(np.sum(probs)), 1e-15)
    if predictor["kind"] == "poisson":
        probs, _lambdas, _feats = predict_score_probs(
            predictor["model"], predictor["scaler"], predictor["cfg"],
            tracker, history, team1, team2, stage,
        )
        return probs

    feats = tracker.snapshot_features(
        team1, team2, pd.Timestamp("2026-06-11"), stage, 2026,
        history, host_teams={norm_team(t) for t in HOSTS_2026},
    )
    row = pd.DataFrame([feats])
    x = row[predictor["features"]].values
    if predictor["scaler"] is not None:
        x = predictor["scaler"].transform(x)
    if predictor["kind"] == "nn":
        import torch

        predictor["model"].eval()
        with torch.no_grad():
            logits = predictor["model"](torch.tensor(x, dtype=torch.float32))
            probs_a = torch.softmax(logits, dim=1).numpy()[0]
    else:
        probs_a = _full_proba(predictor["model"], x)[0]
    probs_a = _redistribute_knockout(probs_a.reshape(1, -1), np.array([stage]))[0]
    if feats["team_a"] == team1:
        probs = probs_a
    else:
        probs = np.array([probs_a[2], probs_a[1], probs_a[0]])
    return probs / max(float(np.sum(probs)), 1e-15)


def calibrated_score_grid(lam_a: float, lam_b: float, cfg: dict, target_probs: np.ndarray) -> np.ndarray:
    grid = score_grid_from_lambdas(lam_a, lam_b, cfg.get("draw_boost", 0.0))
    target = np.asarray(target_probs, dtype=float).copy()
    target = np.clip(target, 0.0, 1.0)
    target = target / max(float(target.sum()), 1e-15)
    i, j = np.indices(grid.shape)
    masks = [i > j, i == j, i < j]
    out = grid.copy()
    for k, mask in enumerate(masks):
        current = float(out[mask].sum())
        if target[k] <= 0:
            out[mask] = 0.0
        elif current > 0:
            out[mask] *= target[k] / current
    return out / max(float(out.sum()), 1e-15)


def expected_goals_from_grid(grid: np.ndarray) -> tuple[float, float]:
    i, j = np.indices(grid.shape)
    return float((i * grid).sum()), float((j * grid).sum())


def sample_score_from_grid(grid: np.ndarray, rng: np.random.Generator, knockout: bool = False) -> tuple[int, int]:
    flat = grid.ravel()
    idx = int(rng.choice(len(flat), p=flat / max(float(flat.sum()), 1e-15)))
    ga, gb = divmod(idx, grid.shape[1])
    if knockout and ga == gb:
        i, j = np.indices(grid.shape)
        p_a = float(grid[i > j].sum())
        p_b = float(grid[i < j].sum())
        if rng.random() < p_a / max(p_a + p_b, 1e-15):
            ga += 1
        else:
            gb += 1
    return int(ga), int(gb)


def predict_sim_match(
    score_model,
    score_scaler,
    score_cfg: dict,
    tracker: FeatureTracker,
    history: dict,
    team1: str,
    team2: str,
    stage: str,
    outcome_predictor: dict | None = None,
) -> tuple[np.ndarray, tuple[float, float], np.ndarray, dict]:
    poisson_probs, (lam_a, lam_b), feats = predict_score_probs(
        score_model, score_scaler, score_cfg, tracker, history, team1, team2, stage,
    )
    target_probs = poisson_probs
    if outcome_predictor is not None:
        target_probs = predict_outcome_probs(outcome_predictor, tracker, history, team1, team2, stage)
        if stage != "group":
            target_probs = _redistribute_knockout(target_probs.reshape(1, -1), np.array([stage]))[0]
    grid = calibrated_score_grid(lam_a, lam_b, score_cfg, target_probs)
    xg_a, xg_b = expected_goals_from_grid(grid)
    return target_probs, (xg_a, xg_b), grid, feats


def sample_score(lam_a: float, lam_b: float, rng: np.random.Generator, knockout: bool = False) -> tuple[int, int]:
    ga = int(min(rng.poisson(max(lam_a, 0.05)), 9))
    gb = int(min(rng.poisson(max(lam_b, 0.05)), 9))
    if knockout and ga == gb:
        p = lam_a / max(lam_a + lam_b, 1e-9)
        if rng.random() < p:
            ga += 1
        else:
            gb += 1
    return ga, gb


def group_pairs(teams: list[str]) -> list[tuple[str, str]]:
    return [(teams[i], teams[j]) for i in range(len(teams)) for j in range(i + 1, len(teams))]


def rank_group(records: dict[str, dict]) -> list[dict]:
    return sorted(
        records.values(),
        key=lambda r: (r["pts"], r["gd"], r["gf"], r["strength"]),
        reverse=True,
    )


def simulate_2026(
    model,
    scaler,
    cfg: dict,
    tracker: FeatureTracker,
    history: dict,
    n_sims: int = 5000,
    outcome_predictor: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups_raw = pd.read_csv(GROUPS_2026)
    groups = {
        g: [norm_team(t) for t in sub["team"].tolist()]
        for g, sub in groups_raw.groupby("group", sort=True)
    }
    base_strength = {t: tracker.ratings[norm_team(t)] for teams in groups.values() for t in teams}
    rng = np.random.default_rng(RNG_SEED)
    stage_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    group_adv_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    match_cache: dict[tuple[str, str, str], tuple[np.ndarray, tuple[float, float], np.ndarray, dict]] = {}

    def get_match(t1: str, t2: str, stage: str):
        key = (norm_team(t1), norm_team(t2), stage)
        if key not in match_cache:
            match_cache[key] = predict_sim_match(
                model, scaler, cfg, tracker, history, t1, t2, stage, outcome_predictor
            )
        return match_cache[key]

    deterministic_rows = deterministic_projection(model, scaler, cfg, tracker, history, groups, outcome_predictor)

    for _ in range(n_sims):
        qualifiers = []
        thirds = []
        for group, teams in groups.items():
            table = {
                t: {"team": t, "group": group, "pts": 0, "gf": 0, "ga": 0, "gd": 0, "strength": base_strength[t]}
                for t in teams
            }
            for t1, t2 in group_pairs(teams):
                _p, _xg, grid, _f = get_match(t1, t2, "group")
                ga, gb = sample_score_from_grid(grid, rng, knockout=False)
                table[t1]["gf"] += ga
                table[t1]["ga"] += gb
                table[t2]["gf"] += gb
                table[t2]["ga"] += ga
                if ga > gb:
                    table[t1]["pts"] += 3
                elif gb > ga:
                    table[t2]["pts"] += 3
                else:
                    table[t1]["pts"] += 1
                    table[t2]["pts"] += 1
            for rec in table.values():
                rec["gd"] = rec["gf"] - rec["ga"]
            ranked = rank_group(table)
            for pos, rec in enumerate(ranked, 1):
                rec = dict(rec)
                rec["group_pos"] = pos
                if pos <= 2:
                    qualifiers.append(rec)
                    group_adv_counts[(group, rec["team"])]["top2"] += 1
                elif pos == 3:
                    thirds.append(rec)
        best_thirds = sorted(thirds, key=lambda r: (r["pts"], r["gd"], r["gf"], r["strength"]), reverse=True)[:8]
        qualifiers.extend(best_thirds)
        for rec in best_thirds:
            group_adv_counts[(rec["group"], rec["team"])]["third"] += 1
        for rec in qualifiers:
            stage_counts[rec["team"]]["R32"] += 1
        bracket = sorted(qualifiers, key=lambda r: (r["group_pos"], -r["pts"], -r["gd"], -r["gf"], -r["strength"]))
        # Convert to high-vs-low by strength after group qualification. This is a
        # simplified deterministic bracket, documented in the report summary.
        bracket = sorted(bracket, key=lambda r: (r["pts"], r["gd"], r["gf"], r["strength"]), reverse=True)
        current = [r["team"] for r in bracket]
        round_names = [("R16", 16), ("QF", 8), ("SF", 4), ("F", 2), ("Champion", 1)]
        while len(current) > 1:
            winners = []
            match_stage = {32: "R32", 16: "R16", 8: "QF", 4: "SF", 2: "F"}[len(current)]
            round_label = {32: "R16", 16: "QF", 8: "SF", 4: "F", 2: "Champion"}[len(current)]
            for i in range(len(current) // 2):
                a = current[i]
                b = current[-(i + 1)]
                _p, _xg, grid, _f = get_match(a, b, match_stage)
                ga, gb = sample_score_from_grid(grid, rng, knockout=True)
                winner = a if ga > gb else b
                winners.append(winner)
                stage_counts[winner][round_label] += 1
            current = winners
    teams = sorted({t for teams in groups.values() for t in teams})
    stage_rows = []
    for team in teams:
        rec = {"team": display_team(team)}
        for stage in ["R32", "R16", "QF", "SF", "F", "Champion"]:
            rec[f"p_{stage}"] = stage_counts[team][stage] / n_sims
        stage_rows.append(rec)
    stage_df = pd.DataFrame(stage_rows).sort_values("p_Champion", ascending=False)
    champ_df = stage_df[["team", "p_Champion"]].rename(columns={"p_Champion": "champion_probability"})
    group_rows = []
    for (group, team), vals in group_adv_counts.items():
        group_rows.append({
            "group": group,
            "team": display_team(team),
            "p_top2": vals["top2"] / n_sims,
            "p_best_third": vals["third"] / n_sims,
            "p_advance": (vals["top2"] + vals["third"]) / n_sims,
        })
    group_df = pd.DataFrame(group_rows).sort_values(["group", "p_advance"], ascending=[True, False])
    return champ_df, stage_df, group_df, deterministic_rows


def deterministic_projection(
    model,
    scaler,
    cfg: dict,
    tracker: FeatureTracker,
    history: dict,
    groups: dict[str, list[str]],
    outcome_predictor: dict | None = None,
) -> pd.DataFrame:
    rows = []
    qualifiers = []
    thirds = []
    for group, teams in groups.items():
        table = {
            t: {"team": t, "group": group, "pts": 0.0, "gf": 0.0, "ga": 0.0, "gd": 0.0, "strength": tracker.ratings[t]}
            for t in teams
        }
        for t1, t2 in group_pairs(teams):
            probs, (la, lb), _grid, _ = predict_sim_match(
                model, scaler, cfg, tracker, history, t1, t2, "group", outcome_predictor
            )
            # Expected-points projection.
            table[t1]["pts"] += 3 * probs[0] + probs[1]
            table[t2]["pts"] += 3 * probs[2] + probs[1]
            table[t1]["gf"] += la
            table[t1]["ga"] += lb
            table[t2]["gf"] += lb
            table[t2]["ga"] += la
            rows.append({
                "stage": "group",
                "group": group,
                "team_a": display_team(t1),
                "team_b": display_team(t2),
                "p_a_win": probs[0],
                "p_draw": probs[1],
                "p_b_win": probs[2],
                "expected_goals_a": la,
                "expected_goals_b": lb,
                "projected_winner": display_team(t1 if probs[0] >= probs[2] else t2),
            })
        for rec in table.values():
            rec["gd"] = rec["gf"] - rec["ga"]
        ranked = rank_group(table)
        for pos, rec in enumerate(ranked, 1):
            rec = dict(rec)
            rec["group_pos"] = pos
            if pos <= 2:
                qualifiers.append(rec)
            elif pos == 3:
                thirds.append(rec)
    qualifiers.extend(sorted(thirds, key=lambda r: (r["pts"], r["gd"], r["gf"], r["strength"]), reverse=True)[:8])
    current = [r["team"] for r in sorted(qualifiers, key=lambda r: (r["pts"], r["gd"], r["gf"], r["strength"]), reverse=True)]
    while len(current) > 1:
        round_label = {32: "R32", 16: "R16", 8: "QF", 4: "SF", 2: "F"}[len(current)]
        winners = []
        for i in range(len(current) // 2):
            a, b = current[i], current[-(i + 1)]
            probs, (la, lb), _grid, _ = predict_sim_match(
                model, scaler, cfg, tracker, history, a, b, round_label, outcome_predictor
            )
            pa = probs[0]
            pb = probs[2]
            winner = a if pa >= pb else b
            winners.append(winner)
            rows.append({
                "stage": round_label,
                "group": "",
                "team_a": display_team(a),
                "team_b": display_team(b),
                "p_a_win": pa,
                "p_draw": probs[1],
                "p_b_win": pb,
                "expected_goals_a": la,
                "expected_goals_b": lb,
                "projected_winner": display_team(winner),
            })
        current = winners
    return pd.DataFrame(rows)


def write_2026_team_diagnostics(
    tracker: FeatureTracker,
    history: dict,
    champ: pd.DataFrame,
    group_adv: pd.DataFrame,
    simulator_name: str,
) -> pd.DataFrame:
    groups_raw = pd.read_csv(GROUPS_2026)
    champ_lookup = {norm_team(r["team"]): float(r["champion_probability"]) for _, r in champ.iterrows()}
    adv_lookup = {norm_team(r["team"]): float(r["p_advance"]) for _, r in group_adv.iterrows()}
    rows = []
    for _, r in groups_raw.iterrows():
        team = norm_team(r["team"])
        rec = list(tracker.recent[team])
        cutoff_4y = pd.Timestamp("2026-06-11") - pd.Timedelta(days=365 * 4)
        s5 = FeatureTracker._stats(rec[-5:])
        s10 = FeatureTracker._stats(rec[-10:])
        s4 = FeatureTracker._stats([e for e in rec if e["date"] >= cutoff_4y][-10:])
        group_teams = [norm_team(t) for t in groups_raw[groups_raw["group"] == r["group"]]["team"]]
        opp_elos = [tracker.ratings[t] for t in group_teams if t != team]
        rows.append({
            "team": display_team(team),
            "group": r["group"],
            "elo": tracker.ratings[team],
            "elo_rank": np.nan,
            "wc_history": wc_history_score(history, team, 2026),
            "wr5": s5["wr"],
            "gd5": s5["gd"],
            "gs5": s5["gs"],
            "cs5": s5["cs"],
            "wr10": s10["wr"],
            "gd10": s10["gd"],
            "wr4y": s4["wr"],
            "gd4y": s4["gd"],
            "avg_group_opp_elo": float(np.mean(opp_elos)) if opp_elos else np.nan,
            "group_advance_probability": adv_lookup.get(team, 0.0),
            "champion_probability": champ_lookup.get(team, 0.0),
        })
    diag = pd.DataFrame(rows)
    diag["elo_rank"] = diag["elo"].rank(ascending=False, method="min").astype(int)
    diag["champion_rank"] = diag["champion_probability"].rank(ascending=False, method="min").astype(int)
    diag = diag.sort_values("champion_probability", ascending=False)
    diag.to_csv(OUT / "team_2026_model_inputs.csv", index=False)

    spain = diag[diag["team"] == "Spain"].iloc[0]
    lines = [
        "# 2026 Forecast Diagnostics",
        "",
        "## Knockout Draw Policy",
        "",
        "All knockout-stage probability vectors are post-processed so P(draw)=0 and win/loss probabilities are renormalized. The calibrated score simulator samples from decisive score cells for knockout matches, with a fallback stochastic tiebreak if numerical rounding ever produces a tied score.",
        "",
        "## Primary Forecast Simulator",
        "",
        f"Primary simulator: {simulator_name}. Match outcome probabilities come from the selected W/D/L model, while the Poisson score layer supplies calibrated scorelines for group goal-difference and goals-for tiebreakers.",
        "",
        "## Spain Gap Against Betting Markets",
        "",
        f"Model Spain champion probability: {spain['champion_probability']:.4f}.",
        f"Model Spain champion rank: {int(spain['champion_rank'])}.",
        f"Model Spain Elo rank in configured field: {int(spain['elo_rank'])}.",
        f"Model Spain group advancement probability: {spain['group_advance_probability']:.4f}.",
        "",
        "The final 2026-only forecast now updates Elo, the outcome model, and the Poisson score layer with supplemental international matches from 2022 through March 2026. Any remaining gap to market-implied probabilities should be interpreted as a limitation of the simplified knockout bracket and missing squad/injury/current-depth information rather than a claim that markets are wrong.",
    ]
    (OUT / "forecast_diagnostics.md").write_text("\n".join(lines))
    return diag


def write_report_summary(
    comp: pd.DataFrame,
    champ: pd.DataFrame,
    stage_df: pd.DataFrame,
    group_df: pd.DataFrame,
    forecast_summary: pd.DataFrame | None = None,
) -> None:
    best = comp.iloc[0]
    best_acc = comp.sort_values(["Test Acc", "Test LL"], ascending=[False, True]).iloc[0]
    lines = [
        "# Final Pipeline Report Summary",
        "",
        "## Methodological Upgrade",
        "",
        "The final pipeline trains on every available pre-tournament supervised match: World Cup rows plus international rows. World Cup examples receive higher sample weight, while all rows receive time-decay weighting so recent matches matter more.",
        "",
        "Validation remains leakage-resistant: hyperparameters are selected with every feasible World Cup checkpoint before 2022. Each checkpoint trains on matches before that tournament and validates on that next World Cup. The 2022 World Cup is the held-out final test.",
        "",
        "## Best 2022 Model",
        "",
        f"Best by 2022 log loss: **{best['Model']}** ({best['Family']}) with test LL **{best['Test LL']}** and accuracy **{best['Test Acc']}**.",
        "",
        f"Best by 2022 accuracy: **{best_acc['Model']}** ({best_acc['Family']}) with accuracy **{best_acc['Test Acc']}** and test LL **{best_acc['Test LL']}**.",
        "",
        "## Model Comparison",
        "",
        comp.to_markdown(index=False),
        "",
        "## 2026 Forecast",
        "",
        "The primary 2026 simulation is a classifier-calibrated score simulator: the best held-out log-loss NN supplies W/D/L probabilities, while the Poisson score layer is reweighted to those probabilities before sampling scorelines for group tiebreakers and knockout advancement. Model selection and 2022 testing use only the historical cache; the final 2026-only fit additionally uses the bundled 2022-2026 supplemental international results to update Elo, form, the outcome model, and the score layer.",
        "",
        "Top champion probabilities:",
        "",
        champ.head(12).to_markdown(index=False),
        "",
        "Forecast variants:",
        "",
        forecast_summary.to_markdown(index=False) if forecast_summary is not None else "",
        "",
        "## Why Models Succeed Or Fail",
        "",
        "- Poisson/Dixon-Coles-style models are useful score generators because football is low scoring, but direct outcome models can provide better W/D/L probabilities.",
        "- MLR tends to be stable because the signal is mostly monotonic team strength and recent form; it can underfit nonlinear effects but avoids extreme probabilities.",
        "- RF captures interactions but can overfit tournament idiosyncrasies when World Cup validation sets are small.",
        "- NN models have the highest capacity and often improve CV on some folds, but they are fragile with only a few hundred World Cup validation matches unless heavily regularized.",
        "- The ensemble is useful only when it is tuned on validation folds and includes genuinely different model families; test-set-selected ensembles should not be reported as final evidence.",
        "",
        "## Bracket Caveat",
        "",
        "The simulator uses the official 12 groups and top-two-plus-eight-best-thirds advancement rule. Because the full FIFA round-of-32 pairing table is awkward to maintain manually, the knockout bracket is seeded high-vs-low by simulated group performance and pre-tournament strength. This is transparent and reproducible, but it is an approximation of the official bracket path.",
    ]
    (OUT / "report_summary.md").write_text("\n".join(lines))


def main() -> None:
    print("[1/6] Loading raw cached matches", flush=True)
    matches = load_raw_matches()
    print(f"      matches: {len(matches)}", flush=True)

    print("[2/6] Building supervised all-match feature table", flush=True)
    df, _tracker_end, _history = build_feature_table(matches)
    df.to_csv(OUT / "features_all_matches.csv", index=False)
    print(
        f"      rows={len(df)} world_cup={int((df.competition == 'world_cup').sum())} "
        f"international={int((df.competition == 'international').sum())}",
        flush=True,
    )

    supplemental_2026 = load_supplemental_2026_matches(matches)
    matches_2026 = sorted(matches + supplemental_2026, key=lambda x: x["date"])
    print(
        f"      2026-only supplement={len(supplemental_2026)} "
        f"latest={max([m['date'] for m in supplemental_2026]).date() if supplemental_2026 else 'n/a'}",
        flush=True,
    )

    specs = [
        ("MLR-original-baseline-5", "MLR", ORIGINAL_BASELINE_FEATS, [{"C": c} for c in [0.01, 0.1, 1.0]], fit_predict_mlr),
        ("MLR-original-18", "MLR", ORIGINAL_DIRECT_18_FEATS, [{"C": c} for c in [0.01, 0.1, 1.0]], fit_predict_mlr),
        ("MLR-original-form", "MLR", ORIGINAL_FORM_FEATS, [{"C": c} for c in [0.01, 0.1, 1.0]], fit_predict_mlr),
        ("MLR-original-matchup", "MLR", ORIGINAL_MATCHUP_FEATS, [{"C": c} for c in [0.01, 0.1, 1.0]], fit_predict_mlr),
        ("MLR-strength", "MLR", STRENGTH_FEATS, [{"C": c} for c in [0.01, 0.1, 1.0]], fit_predict_mlr),
        ("MLR-full", "MLR", FULL_FEATS, [{"C": c} for c in [0.01, 0.1, 1.0]], fit_predict_mlr),
        ("MLR-recent-no-elo", "MLR", NO_ELO_FEATS, [{"C": c} for c in [0.01, 0.1, 1.0]], fit_predict_mlr),
        ("RF-original-18", "RF", ORIGINAL_DIRECT_18_FEATS, [
            {"n_estimators": 250, "max_depth": 5, "min_samples_leaf": 20},
            {"n_estimators": 350, "max_depth": 8, "min_samples_leaf": 10},
        ], fit_predict_rf),
        ("RF-strength", "RF", STRENGTH_FEATS, [
            {"n_estimators": 250, "max_depth": 5, "min_samples_leaf": 20},
            {"n_estimators": 350, "max_depth": 8, "min_samples_leaf": 10},
        ], fit_predict_rf),
        ("RF-full", "RF", FULL_FEATS, [
            {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 20},
            {"n_estimators": 500, "max_depth": 10, "min_samples_leaf": 10},
        ], fit_predict_rf),
        ("RF-recent-no-elo", "RF", NO_ELO_FEATS, [
            {"n_estimators": 300, "max_depth": 5, "min_samples_leaf": 20},
            {"n_estimators": 500, "max_depth": 9, "min_samples_leaf": 10},
        ], fit_predict_rf),
        ("Poisson-independent", "Poisson", SCORE_FEATS, [
            {"alpha": a, "draw_boost": 0.0} for a in [0.001, 0.01, 0.1]
        ], fit_predict_poisson),
        ("Poisson-draw-inflated", "Poisson", SCORE_FEATS, [
            {"alpha": a, "draw_boost": b}
            for a in [0.001, 0.01, 0.1]
            for b in [0.15, 0.35, 0.55, 0.75]
        ], fit_predict_poisson),
        ("NN-strength", "NN", STRENGTH_FEATS, [
            {"hidden1": 24, "hidden2": 12, "dropout": 0.25, "lr": 0.005, "weight_decay": 1e-3},
            {"hidden1": 32, "hidden2": 16, "dropout": 0.35, "lr": 0.003, "weight_decay": 1e-3},
        ], fit_predict_nn),
        ("NN-original-18", "NN", ORIGINAL_DIRECT_18_FEATS, [
            {"hidden1": 24, "hidden2": 12, "dropout": 0.30, "lr": 0.004, "weight_decay": 1e-3},
            {"hidden1": 32, "hidden2": 16, "dropout": 0.40, "lr": 0.003, "weight_decay": 2e-3},
        ], fit_predict_nn),
        ("NN-full-regularized", "NN", FULL_FEATS, [
            {"hidden1": 32, "hidden2": 16, "dropout": 0.40, "lr": 0.003, "weight_decay": 2e-3},
            {"hidden1": 48, "hidden2": 16, "dropout": 0.50, "lr": 0.002, "weight_decay": 3e-3},
        ], fit_predict_nn),
        ("NN-expanded-no-underdog", "NN", FULL_FEATS_NO_UNDERDOG, [
            {"hidden1": 32, "hidden2": 16, "dropout": 0.40, "lr": 0.003, "weight_decay": 2e-3},
            {"hidden1": 48, "hidden2": 16, "dropout": 0.50, "lr": 0.002, "weight_decay": 3e-3},
        ], fit_predict_nn),
        ("NN-recent-no-elo", "NN", NO_ELO_FEATS, [
            {"hidden1": 24, "hidden2": 12, "dropout": 0.35, "lr": 0.003, "weight_decay": 2e-3},
        ], fit_predict_nn),
    ]

    print("[3/6] Training and evaluating model families", flush=True)
    results: list[ModelResult] = []
    for spec in specs:
        results.append(evaluate_spec(df, *spec))

    train_final, test_2022 = final_2022_split(df)
    ensemble = tune_ensemble(results, test_2022["label"].values)
    results.append(ensemble)

    print("[4/6] Writing model artifacts", flush=True)
    comp = comparison_table(results)
    comp.to_csv(OUT / "model_comparison.csv", index=False)
    (OUT / "model_comparison.md").write_text(comp.to_markdown(index=False))
    dataset_summary = write_dataset_summary(df)
    feature_usage = write_feature_artifacts(results)
    fold_metrics = write_fold_metrics(results, df)
    with open(OUT / "confusions.txt", "w") as f:
        for m in sorted(results, key=lambda x: x.test_ll):
            f.write(f"\n=== {m.name} ===\n")
            f.write(confusion_text(m.test_cm) + "\n")
    with open(OUT / "ensemble_weights.json", "w") as f:
        json.dump(ensemble.best_config, f, indent=2)
    for m in results:
        if m.family != "Ensemble":
            slug = m.name.replace(" ", "_").replace("/", "_")
            family_dir = MODELS_DIR / m.family.lower()
            family_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump({
                "name": m.name,
                "family": m.family,
                "features": m.features,
                "config": m.best_config,
                "model": m.fitted,
                "scaler": m.scaler,
            }, family_dir / f"{slug}.joblib")

    print("[5/6] Simulating 2026 World Cup", flush=True)
    best_score = min([m for m in results if m.family == "Poisson"], key=lambda m: m.mean_cv_ll)
    df_2026, _tracker_augmented, _history_augmented = build_feature_table(matches_2026, history_cutoff_year=2026)
    df_2026.to_csv(OUT / "features_all_matches_2026_augmented.csv", index=False)
    write_2026_training_summary(df, df_2026, supplemental_2026)
    score_model, score_scaler, score_cfg = fit_final_2026_score_model(df_2026, best_score)
    tracker_2026, history_2026 = tracker_after_matches(matches_2026, history_cutoff_year=2026)
    results_by_name = {m.name: m for m in results}
    predictor_cache: dict[str, dict] = {}
    forecast_specs = [
        ("poisson_only", "Poisson-only score baseline", None),
        ("nn_original_18_hybrid", "NN-original-18 calibrated score simulator", results_by_name["NN-original-18"]),
        ("nn_expanded_no_underdog_hybrid", "NN-expanded-no-underdog calibrated score simulator", results_by_name["NN-expanded-no-underdog"]),
        ("ensemble_hybrid", "Validation-tuned ensemble calibrated score simulator", ensemble),
    ]
    forecast_outputs = {}
    forecast_summary_rows = []
    for slug, label, outcome_result in forecast_specs:
        print(f"      forecast variant: {label}", flush=True)
        outcome_predictor = None
        if outcome_result is not None:
            outcome_predictor = fit_final_2026_outcome_predictor(
                df_2026, outcome_result, results_by_name, predictor_cache
            )
        v_champ, v_stage, v_group, v_path = simulate_2026(
            score_model, score_scaler, score_cfg, tracker_2026, history_2026,
            n_sims=5000, outcome_predictor=outcome_predictor,
        )
        v_champ.to_csv(FORECAST_DIR / f"champion_probabilities_2026_{slug}.csv", index=False)
        v_stage.to_csv(FORECAST_DIR / f"team_stage_probabilities_2026_{slug}.csv", index=False)
        v_group.to_csv(FORECAST_DIR / f"group_advancement_2026_{slug}.csv", index=False)
        v_path.to_csv(FORECAST_DIR / f"most_likely_2026_path_{slug}.csv", index=False)
        spain_prob = float(v_champ.loc[v_champ["team"] == "Spain", "champion_probability"].iloc[0])
        top = v_champ.iloc[0]
        forecast_summary_rows.append({
            "Forecast": label,
            "Top Team": top["team"],
            "Top Champion Probability": float(top["champion_probability"]),
            "Spain Champion Probability": spain_prob,
        })
        forecast_outputs[slug] = (label, v_champ, v_stage, v_group, v_path)

    forecast_summary = pd.DataFrame(forecast_summary_rows)
    forecast_summary.to_csv(OUT / "forecast_variant_summary.csv", index=False)
    primary_slug = "nn_original_18_hybrid"
    primary_label, champ, stage_probs, group_adv, path = forecast_outputs[primary_slug]
    champ.to_csv(OUT / "champion_probabilities_2026.csv", index=False)
    stage_probs.to_csv(OUT / "team_stage_probabilities_2026.csv", index=False)
    group_adv.to_csv(OUT / "group_advancement_2026.csv", index=False)
    path.to_csv(OUT / "most_likely_2026_path.csv", index=False)
    team_diag = write_2026_team_diagnostics(tracker_2026, history_2026, champ, group_adv, primary_label)

    print("[6/6] Writing report summary", flush=True)
    write_figures(comp, fold_metrics, feature_usage, champ, group_adv, df)
    write_report_summary(comp, champ, stage_probs, group_adv, forecast_summary)
    print("\nTop models:")
    print(comp.head(12).to_string(index=False))
    print("\nTop 2026 champions:")
    print(champ.head(12).to_string(index=False))
    print(f"\nArtifacts written to {OUT}")


if __name__ == "__main__":
    main()
