//! Exact finite-graph WDL oracle. Unknown closed components are draws.
use crate::Result;
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Wdl {
    Loss,
    Draw,
    Win,
}
pub fn solve(edges: &[Vec<usize>], terminal: &[Option<Wdl>]) -> Result<Vec<Wdl>> {
    if edges.len() != terminal.len() {
        return Err("terminal count".into());
    }
    let n = edges.len();
    let mut predecessors = vec![Vec::new(); n];
    let mut remaining = vec![0usize; n];
    for (v, successors) in edges.iter().enumerate() {
        let mut unique = successors.clone();
        unique.sort_unstable();
        unique.dedup();
        for &s in &unique {
            if s >= n {
                return Err("edge out of graph".into());
            }
            predecessors[s].push(v);
        }
        remaining[v] = unique.len();
        if unique.is_empty() && terminal[v].is_none() {
            return Err("empty node requires checkmate/stalemate seed".into());
        }
    }
    let mut values = terminal.to_vec();
    let mut queue = VecDeque::new();
    for (v, value) in values.iter().enumerate() {
        if matches!(value, Some(Wdl::Win | Wdl::Loss)) {
            queue.push_back(v);
        }
    }
    while let Some(s) = queue.pop_front() {
        for &p in &predecessors[s] {
            if values[p].is_some() {
                continue;
            }
            if values[s] == Some(Wdl::Loss) {
                values[p] = Some(Wdl::Win);
                queue.push_back(p);
            } else {
                remaining[p] -= 1;
                if remaining[p] == 0 {
                    values[p] = Some(Wdl::Loss);
                    queue.push_back(p);
                }
            }
        }
    }
    Ok(values
        .into_iter()
        .map(|value| value.unwrap_or(Wdl::Draw))
        .collect())
}
/// Dense local stencil propagation: two bitplanes and one validity mask.
/// Rows are independent; masks prevent missing moves from counting as losses.
pub fn update_row(
    unknown: bool,
    losing_children: u64,
    winning_children: u64,
    legal: u64,
) -> Option<Wdl> {
    if !unknown {
        return None;
    }
    if losing_children & legal != 0 {
        Some(Wdl::Win)
    } else if legal != 0 && winning_children & legal == legal {
        Some(Wdl::Loss)
    } else {
        None
    }
}
