//! The forced-win walk's knobs: what a skipped fight costs, read from a TOML
//! file because the numbers exist to be tweaked.
//!
//! A forced-win batch walks runs without playing their combats — every fight
//! is harvested at its entry and resolved as won — so the config is what
//! stands in for the fight's price: an HP loss drawn per tier and scaled by
//! act, and a chance the fight would have drunk a potion. The file is
//! refused whole on any field it does not know or bound it does not honor,
//! and the parsed config is echoed into the bank's manifest, so a bank
//! always names the distribution that shaped it.

use std::path::Path;

use sts2_rng::MegaRandom;

use crate::library::Tier;

/// A loaded, validated forced-win config.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForceWins {
    /// The HP loss a skipped fight costs, per tier, as fractions of max HP.
    pub hp_loss: PerTier<Span>,
    /// Multiplier on the drawn loss, indexed by act; an act past the end
    /// keeps the last entry.
    pub act_scale: Vec<f64>,
    /// The chance a skipped fight consumes one held potion, per tier.
    pub potion_use: PerTier<f64>,
    /// The injected loss never leaves the player below this.
    pub min_hp: i32,
}

/// One value for each room tier.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerTier<T> {
    pub hallway: T,
    pub elite: T,
    pub boss: T,
}

impl<T> PerTier<T> {
    fn of(&self, tier: Tier) -> &T {
        match tier {
            Tier::Hallway => &self.hallway,
            Tier::Elite => &self.elite,
            Tier::Boss => &self.boss,
        }
    }
}

/// A uniform draw's bounds, both inclusive.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Span {
    pub min: f64,
    pub max: f64,
}

/// A config that could not be read, parsed, or honored, and why.
#[derive(Debug)]
pub struct ConfigError(pub String);

impl std::fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

impl std::error::Error for ConfigError {}

impl ForceWins {
    /// Reads and validates a config file, refusing it whole on the first
    /// unknown field or dishonored bound.
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path)
            .map_err(|error| ConfigError(format!("{}: {error}", path.display())))?;
        let config: Self = toml::from_str(&text)
            .map_err(|error| ConfigError(format!("{}: {error}", path.display())))?;
        for tier in Tier::ALL {
            let span = config.hp_loss.of(tier);
            if !(0.0..=1.0).contains(&span.min) || !(span.min..=1.0).contains(&span.max) {
                return Err(ConfigError(format!(
                    "hp_loss.{tier} wants 0 <= min <= max <= 1, not {} ..= {}",
                    span.min, span.max
                )));
            }
            let chance = *config.potion_use.of(tier);
            if !(0.0..=1.0).contains(&chance) {
                return Err(ConfigError(format!(
                    "potion_use.{tier} is a chance, so 0 <= {chance} <= 1"
                )));
            }
        }
        if config.act_scale.is_empty() || config.act_scale.iter().any(|scale| *scale < 0.0) {
            return Err(ConfigError(
                "act_scale wants at least one non-negative multiplier".into(),
            ));
        }
        if config.min_hp < 1 {
            return Err(ConfigError(format!(
                "min_hp keeps the player alive, so it is at least 1, not {}",
                config.min_hp
            )));
        }
        Ok(config)
    }

    /// The hit points the player leaves a skipped fight with: the drawn,
    /// act-scaled loss off their current hit points, never below `min_hp`
    /// and never above where they stood.
    #[must_use]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "hit points are far below f64 precision"
    )]
    pub fn hp_after(
        &self,
        tier: Tier,
        act: usize,
        max_hp: i32,
        current_hp: i32,
        rng: &mut MegaRandom,
    ) -> i32 {
        let span = self.hp_loss.of(tier);
        let fraction = span.min + unit(rng) * (span.max - span.min);
        let scale = self
            .act_scale
            .get(act)
            .or(self.act_scale.last())
            .copied()
            .unwrap_or(1.0);
        let loss = (f64::from(max_hp) * fraction * scale).round() as i32;
        (current_hp - loss).max(self.min_hp).min(current_hp)
    }

    /// Whether this skipped fight consumes one held potion.
    #[must_use]
    pub fn uses_potion(&self, tier: Tier, rng: &mut MegaRandom) -> bool {
        unit(rng) < *self.potion_use.of(tier)
    }

    /// The parsed config as a manifest annotation, so the bank it shaped
    /// names it.
    #[must_use]
    pub fn echo(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("the config serializes")
    }
}

/// A uniform draw in `[0, 1)`, off the top 53 bits.
#[allow(
    clippy::cast_precision_loss,
    reason = "53 bits is exactly what an f64 mantissa holds"
)]
fn unit(rng: &mut MegaRandom) -> f64 {
    (rng.next_u64() >> 11) as f64 / (1_u64 << 53) as f64
}
