# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional


def aggregate_turn_scores(
    turn_scores: Any,
    *,
    method: str = "mean",
    weights: Optional[list[float]] = None,
    min_floor: Optional[float] = None,
) -> Optional[float]:
    """Aggregate multi-turn reward scores into a single scalar.

    Supported methods:
    - "mean": simple average (recommended for eval-aligned training)
    - "linear": linearly increasing weights (DEPRECATED: causes last-turn overfitting)
    - "soft_linear": gentler version with base weight to prevent early-turn neglect
    - "min_mean": mean with minimum constraint (all turns must be decent)
    - "last": last valid score only
    - custom weights: pass `weights` matching the number of valid scores
    
    Args:
        turn_scores: list of per-turn scores or single scalar
        method: aggregation method name
        weights: custom weights (overrides method)
        min_floor: if set, final score = max(min_floor, min(scores)) * alpha + aggregated * (1-alpha)
    """
    if isinstance(turn_scores, (int, float)):
        return float(turn_scores)
    if not isinstance(turn_scores, (list, tuple)):
        return None

    valid_values: list[float] = []
    for item in turn_scores:
        if item is None:
            continue
        if isinstance(item, (int, float)):
            valid_values.append(float(item))
        else:
            try:
                valid_values.append(float(item))
            except (TypeError, ValueError):
                continue

    if not valid_values:
        return None
    if len(valid_values) == 1:
        return valid_values[0]

    if weights:
        if len(weights) != len(valid_values):
            return None
        weighted_sum = sum(w * v for w, v in zip(weights, valid_values))
        total_w = sum(weights)
        return weighted_sum / total_w if total_w > 0 else None

    if method == "last":
        return valid_values[-1]
    
    if method == "mean":
        return sum(valid_values) / len(valid_values)
    
    if method == "linear":
        # DEPRECATED: causes models to only optimize last turn
        # Linearly increasing weights: [1, 2, 3, ..., n]
        lin_weights = [i + 1 for i in range(len(valid_values))]
        weighted_sum = sum(w * v for w, v in zip(lin_weights, valid_values))
        return weighted_sum / sum(lin_weights)
    
    if method == "soft_linear":
        # Gentler version: base_weight + small increment
        # e.g., for 3 turns: [1.0, 1.1, 1.2] instead of [1, 2, 3]
        # This prevents neglecting early turns while still rewarding consistency
        n = len(valid_values)
        base_weight = 1.0
        increment = 0.1  # Small increment per turn
        soft_weights = [base_weight + i * increment for i in range(n)]
        weighted_sum = sum(w * v for w, v in zip(soft_weights, valid_values))
        return weighted_sum / sum(soft_weights)
    
    if method == "min_mean":
        # Combination: penalize if any turn is bad, reward overall consistency
        # final = 0.3 * min(scores) + 0.7 * mean(scores)
        # This forces model to care about ALL turns
        min_score = min(valid_values)
        mean_score = sum(valid_values) / len(valid_values)
        alpha = 0.3  # Weight for minimum score constraint
        return alpha * min_score + (1 - alpha) * mean_score
    
    if method == "harmonic":
        # Harmonic mean: heavily penalizes low outliers
        # Good for ensuring no turn is neglected
        try:
            return len(valid_values) / sum(1.0 / max(v, 0.01) for v in valid_values)
        except ZeroDivisionError:
            return 0.0

    return None


def _weighted_average_numeric(val: Any) -> Optional[float]:
    """Backward-compatible helper for multi-turn aggregation (mean weighting)."""
    return aggregate_turn_scores(val, method="mean")


def _pick_last_numeric(val: Any) -> Optional[float]:
    """Alias for backward compatibility. Now uses weighted average."""
    return _weighted_average_numeric(val)


def _extract_interaction_reward(
    extra_info: Optional[dict],
    aggregation_method: str = "min_mean",
) -> Optional[float]:
    """Extract and aggregate multi-turn interaction rewards.
    
    Args:
        extra_info: dict containing turn scores from interaction
        aggregation_method: how to aggregate turn scores. Options:
            - "min_mean": 0.3*min + 0.7*mean (recommended, prevents any-turn neglect)
            - "mean": simple average (baseline, matches typical eval)
            - "soft_linear": gentle recency bias without extreme last-turn focus
            - "harmonic": heavily penalizes low outliers
            - "linear": DEPRECATED (causes last-turn overfitting)
    
    Returns:
        Aggregated reward score or None if not found
    """
    if not isinstance(extra_info, dict):
        return None
    
    for key in ("turn_scores", "interaction_reward", "interaction_score"):
        if key in extra_info:
            return aggregate_turn_scores(extra_info.get(key), method=aggregation_method)

    tool_extra = extra_info.get("tool_extra_fields")
    if isinstance(tool_extra, dict):
        for key in ("turn_scores", "interaction_reward", "interaction_score"):
            if key in tool_extra:
                return aggregate_turn_scores(tool_extra.get(key), method=aggregation_method)

    rollout_scores = extra_info.get("rollout_reward_scores")
    if isinstance(rollout_scores, dict):
        for key in ("turn_scores", "interaction_reward", "interaction_score"):
            if key in rollout_scores:
                return aggregate_turn_scores(rollout_scores.get(key), method=aggregation_method)

    return None


def compute_score(solution_str, ground_truth, extra_info=None, **kwargs):
    """Return interaction reward for STEER/SaFeR-Steer.

    The interaction reward is expected to be stored in extra_info (e.g., tool_extra_fields.turn_scores).
    If missing, returns 0.0.
    
    Kwargs:
        aggregation_method: str, how to aggregate turn scores. Default "min_mean".
            Options: "min_mean", "mean", "soft_linear", "harmonic"
            Avoid "linear" as it causes last-turn overfitting.
    """
    aggregation_method = kwargs.get("aggregation_method", "min_mean")
    score = _extract_interaction_reward(extra_info, aggregation_method=aggregation_method)
    if score is None:
        return 0.0
    return float(score)
