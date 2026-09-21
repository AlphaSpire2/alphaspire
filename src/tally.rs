//! The loss table a batch prints when it is done: what the batch lost to,
//! costliest first, with the floor span each cost was paid over.
//!
//! Two batches keep one: [`crate::selfplay::OutcomeTally`] counts runs and
//! the killers that ended them, [`crate::library::FightTally`] counts
//! playouts and the encounters that won them. They aggregate different
//! things and share a table, so the two epilogues read the same and stay
//! that way.

use std::collections::BTreeMap;
use std::fmt::Display;

/// One row's tally: how many times the row was hit, and where the floors it
/// was hit on start and stop.
#[derive(Clone, Debug)]
struct Spread {
    count: usize,
    lowest: u32,
    highest: u32,
}

/// Rows keyed by act index, `Label` — a room family, a fight tier — and
/// name, in a sorted map so equal counts print in one stable order.
#[derive(Clone, Debug)]
pub(crate) struct Rows<Label: Ord>(BTreeMap<(usize, Label, String), Spread>);

impl<Label: Ord> Default for Rows<Label> {
    fn default() -> Self {
        Self(BTreeMap::new())
    }
}

impl<Label: Ord + Display> Rows<Label> {
    /// Counts one hit against its row.
    pub(crate) fn record(&mut self, act: usize, label: Label, name: String, floor: u32) {
        let row = self.0.entry((act, label, name)).or_insert(Spread {
            count: 0,
            lowest: floor,
            highest: floor,
        });
        row.count += 1;
        row.lowest = row.lowest.min(floor);
        row.highest = row.highest.max(floor);
    }

    /// The table, costliest row first, at most `limit` rows plus a
    /// remainder line naming what the limit cut. `noun` is what the counts
    /// are of, for that remainder line.
    pub(crate) fn table(&self, limit: usize, noun: &str) -> Vec<String> {
        let mut rows: Vec<_> = self.0.iter().collect();
        rows.sort_by(|left, right| right.1.count.cmp(&left.1.count).then(left.0.cmp(right.0)));
        let mut lines: Vec<String> = rows
            .iter()
            .take(limit)
            .map(|((act, label, name), row)| {
                let floors = if row.lowest == row.highest {
                    format!("floor {}", row.lowest)
                } else {
                    format!("floors {}-{}", row.lowest, row.highest)
                };
                format!("{:>4}  {name} ({label}, act {act}, {floors})", row.count)
            })
            .collect();
        if rows.len() > limit {
            let remaining: usize = rows[limit..].iter().map(|(_, row)| row.count).sum();
            lines.push(format!(
                "      … and {remaining} {noun} over {} more rows",
                rows.len() - limit
            ));
        }
        lines
    }
}
