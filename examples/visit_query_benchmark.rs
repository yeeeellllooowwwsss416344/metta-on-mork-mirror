/// Phase 1 Validation Benchmark: Space::query() vs Space::visit_query()
///
/// Tests the public Space trait APIs via `&dyn Space` (not MorkSpace internals).
/// Q6 workload: Cartesian product of Automotive companies.
///
/// Modes:
///   query      — Space::query(), measure time + result count
///   visit      — Space::visit_query() consume-all, measure time + count
///   count_match — verify query().len() == visit_query(counter) for small N
///   early_stop  — verify callback contract: visited=100, return false, kernel stops
///   profile     — visit_query consume-all, print timestamps for external memory monitoring

use hyperon_atom::Atom;
use hyperon_space::{Space, SpaceMut};
use metta_on_mork::MorkSpace;
use std::time::Instant;

fn build_q6() -> Atom {
    Atom::expr([
        Atom::sym(","),
        Atom::expr([Atom::sym("industry"), Atom::var("c1"), Atom::sym("Automotive")]),
        Atom::expr([Atom::sym("industry"), Atom::var("c2"), Atom::sym("Automotive")]),
    ])
}

fn load_dataset(n: usize) -> MorkSpace {
    let mut space = MorkSpace::new();
    let companies: Vec<String> = (0..n / 10).map(|i| format!("company_{i}")).collect();
    let persons: Vec<String> = (0..n / 10).map(|i| format!("person_{i}")).collect();
    let vehicles: Vec<String> = (0..n / 10).map(|i| format!("vehicle_{i}")).collect();

    for c in &companies {
        space.add(Atom::expr([Atom::sym("company"), Atom::sym(c)]));
    }
    for p in &persons {
        space.add(Atom::expr([Atom::sym("person"), Atom::sym(p)]));
    }
    for v in &vehicles {
        space.add(Atom::expr([Atom::sym("vehicle"), Atom::sym(v)]));
    }
    for (i, c) in companies.iter().enumerate() {
        let p = &persons[i % persons.len()];
        space.add(Atom::expr([Atom::sym("founded_by"), Atom::sym(c), Atom::sym(p)]));
        space.add(Atom::expr([Atom::sym("industry"), Atom::sym(c), Atom::sym("Automotive")]));
        if i < vehicles.len() {
            space.add(Atom::expr([Atom::sym("produces"), Atom::sym(c), Atom::sym(&vehicles[i])]));
        }
    }
    for c in &companies {
        space.add(Atom::expr([Atom::sym("isa"), Atom::sym(c), Atom::sym("Company")]));
    }
    for p in &persons {
        space.add(Atom::expr([Atom::sym("isa"), Atom::sym(p), Atom::sym("Person")]));
    }
    space.add(Atom::expr([Atom::sym("isa"), Atom::sym("Company"), Atom::sym("Entity")]));
    space.add(Atom::expr([Atom::sym("isa"), Atom::sym("Person"), Atom::sym("Entity")]));
    for c in &companies {
        space.add(Atom::expr([Atom::sym("source"), Atom::sym(c), Atom::sym("\"Wikipedia\"")]));
        space.add(Atom::expr([Atom::sym("confidence"), Atom::sym(c), Atom::sym("0.98")]));
    }
    space
}

/// Run query via &dyn Space (public trait API)
fn run_query(space: &dyn Space, query: &Atom) -> (usize, u128) {
    let t0 = Instant::now();
    let count = space.query(query).len();
    let elapsed = t0.elapsed().as_millis();
    (count, elapsed)
}

/// Run visit_query via &dyn Space (public trait API)
fn run_visit(space: &dyn Space, query: &Atom) -> (usize, u128) {
    let t0 = Instant::now();
    let count = space.visit_query(query, &mut |_| true);
    let elapsed = t0.elapsed().as_millis();
    (count, elapsed)
}

/// Test 1: Count compatibility — query().len() == visit_query(counter)
fn test_count_match(space: &dyn Space, query: &Atom) -> bool {
    let query_count = space.query(query).len();
    let mut visit_count = 0usize;
    space.visit_query(query, &mut |_| {
        visit_count += 1;
        true
    });
    eprintln!("count_match: query={query_count} visit={visit_count} equal={}", query_count == visit_count);
    query_count == visit_count
}

/// Test 2: Early termination contract
/// callback receives N results, returns false after 100.
/// Kernel must stop. Return value must equal delivered count.
fn test_early_stop(space: &dyn Space, query: &Atom) -> bool {
    let limit = 100usize;
    let mut delivered = 0usize;
    let total = space.visit_query(query, &mut |_| {
        delivered += 1;
        delivered < limit
    });
    let correct = delivered == limit && total == limit;
    eprintln!("early_stop: requested={limit} delivered={delivered} total_returned={total} correct={correct}");
    correct
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("visit");
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(50_000);

    let space = load_dataset(n);
    let q6 = build_q6();
    let atom_count = space.atom_count().unwrap_or(0);

    // All calls go through &dyn Space (public trait API)
    let space_ref: &dyn Space = &space;

    eprintln!("N={n} atoms={atom_count} mode={mode}");

    match mode {
        "query" => {
            let (count, elapsed) = run_query(space_ref, &q6);
            println!("query {n} {count} {elapsed}");
        }
        "visit" => {
            let (count, elapsed) = run_visit(space_ref, &q6);
            println!("visit {n} {count} {elapsed}");
        }
        "count_match" => {
            // For small N only (full query materialization needed)
            let ok = test_count_match(space_ref, &q6);
            println!("count_match {n} {ok}");
        }
        "early_stop" => {
            let ok = test_early_stop(space_ref, &q6);
            println!("early_stop {n} {ok}");
        }
        "profile" => {
            // visit_query with timestamps for external memory monitoring (1s interval)
            let t_start = Instant::now();
            let count = space_ref.visit_query(&q6, &mut |_| true);
            let elapsed = t_start.elapsed().as_millis();
            println!("profile {n} {count} {elapsed}");
        }
        _ => {
            eprintln!("Usage: visit_query_benchmark [query|visit|count_match|early_stop|profile] [N]");
            std::process::exit(1);
        }
    }
}
