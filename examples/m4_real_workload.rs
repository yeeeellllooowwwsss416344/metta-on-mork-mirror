/// M4 Real Workload Test: Semantic Schema Pipeline
///
/// Mode A: Correctness workload (scaling ladder: 10K, 100K, 1M)
/// Mode B: Stress workload (adversarial graph shapes + human exploration)

use hyperon_atom::{Atom, VariableAtom};
use hyperon_space::{Space, SpaceMut};
use metta_on_mork::MorkSpace;
use std::time::Instant;

fn main() {
    println!("=== M4 Real Workload Test ===\n");

    // Mode A: Correctness workload with scaling ladder
    println!("--- Mode A: Correctness Workload ---\n");
    for n in &[10_000, 100_000, 1_000_000] {
        mode_a_correctness(*n);
    }

    // Mode B: Stress workload
    println!("\n--- Mode B: Stress Workload ---\n");
    mode_b_star_graph(100_000);
    mode_b_dense_graph(10_000, 50);
    mode_b_deep_chain(100_000);
    mode_b_human_exploration(100_000);
}

/// Mode A: Semantic schema pipeline correctness
fn mode_a_correctness(n: usize) {
    let start = Instant::now();
    let mut space = MorkSpace::new();

    // Generate entities: company_0, person_0, vehicle_0, etc.
    let companies: Vec<String> = (0..n/10).map(|i| format!("company_{}", i)).collect();
    let persons: Vec<String> = (0..n/10).map(|i| format!("person_{}", i)).collect();
    let vehicles: Vec<String> = (0..n/10).map(|i| format!("vehicle_{}", i)).collect();

    // Load entities
    for c in &companies {
        space.add(Atom::expr([Atom::sym("company"), Atom::sym(c)]));
    }
    for p in &persons {
        space.add(Atom::expr([Atom::sym("person"), Atom::sym(p)]));
    }
    for v in &vehicles {
        space.add(Atom::expr([Atom::sym("vehicle"), Atom::sym(v)]));
    }

    // Load relations (founded_by, industry, produces)
    for (i, c) in companies.iter().enumerate() {
        let p = &persons[i % persons.len()];
        space.add(Atom::expr([Atom::sym("founded_by"), Atom::sym(c), Atom::sym(p)]));
        space.add(Atom::expr([Atom::sym("industry"), Atom::sym(c), Atom::sym("Automotive")]));
        if i < vehicles.len() {
            space.add(Atom::expr([Atom::sym("produces"), Atom::sym(c), Atom::sym(&vehicles[i])]));
        }
    }

    // Load ontology (isa hierarchy)
    for c in &companies {
        space.add(Atom::expr([Atom::sym("isa"), Atom::sym(c), Atom::sym("Company")]));
    }
    for p in &persons {
        space.add(Atom::expr([Atom::sym("isa"), Atom::sym(p), Atom::sym("Person")]));
    }
    space.add(Atom::expr([Atom::sym("isa"), Atom::sym("Company"), Atom::sym("Entity")]));
    space.add(Atom::expr([Atom::sym("isa"), Atom::sym("Person"), Atom::sym("Entity")]));

    // Load metadata
    for c in &companies {
        space.add(Atom::expr([Atom::sym("source"), Atom::sym(c), Atom::sym("\"Wikipedia\"")]));
        space.add(Atom::expr([Atom::sym("confidence"), Atom::sym(c), Atom::sym("0.98")]));
    }

    let load_time = start.elapsed();
    let atom_count = space.atom_count().unwrap_or(0);

    // Query 1: Direct lookup
    let q1_start = Instant::now();
    let q1 = Atom::expr([Atom::sym("company"), Atom::var("x")]);
    let q1_results = space.query(&q1).len();
    let q1_time = q1_start.elapsed();

    // Query 2: Reverse lookup
    let q2_start = Instant::now();
    let q2 = Atom::expr([Atom::sym("founded_by"), Atom::var("person"), Atom::sym("company_0")]);
    let q2_results = space.query(&q2).len();
    let q2_time = q2_start.elapsed();

    // Query 3: Multi-hop (isa chain)
    let q3_start = Instant::now();
    let q3 = Atom::expr([Atom::sym("isa"), Atom::sym("company_0"), Atom::var("type")]);
    let q3_results = space.query(&q3).len();
    let q3_time = q3_start.elapsed();

    // Query 4: Conjunction (company AND Automotive)
    let q4_start = Instant::now();
    let q4 = Atom::expr([
        Atom::sym(","),
        Atom::expr([Atom::sym("company"), Atom::var("x")]),
        Atom::expr([Atom::sym("industry"), Atom::var("x"), Atom::sym("Automotive")]),
    ]);
    let q4_results = space.query(&q4).len();
    let q4_time = q4_start.elapsed();

    // Query 5: Provenance
    let q5_start = Instant::now();
    let q5 = Atom::expr([Atom::sym("source"), Atom::sym("company_0"), Atom::var("src")]);
    let q5_results = space.query(&q5).len();
    let q5_time = q5_start.elapsed();

    // Query 6: Variable joins (companies in same industry)
    let q6_start = Instant::now();
    let q6 = Atom::expr([
        Atom::sym(","),
        Atom::expr([Atom::sym("industry"), Atom::var("c1"), Atom::sym("Automotive")]),
        Atom::expr([Atom::sym("industry"), Atom::var("c2"), Atom::sym("Automotive")]),
    ]);
    let q6_results = space.query(&q6).len();
    let q6_time = q6_start.elapsed();

    println!("N={:>8}: load={:>6}ms, atoms={:>8}", n, load_time.as_millis(), atom_count);
    println!("  Q1 direct:      {:>6} results, {:>3}µs", q1_results, q1_time.as_micros());
    println!("  Q2 reverse:     {:>6} results, {:>3}µs", q2_results, q2_time.as_micros());
    println!("  Q3 multi-hop:   {:>6} results, {:>3}µs", q3_results, q3_time.as_micros());
    println!("  Q4 conjunction:  {:>6} results, {:>3}µs", q4_results, q4_time.as_micros());
    println!("  Q5 provenance:  {:>6} results, {:>3}µs", q5_results, q5_time.as_micros());
    println!("  Q6 var joins:   {:>6} results, {:>3}µs", q6_results, q6_time.as_micros());
    println!();
}

/// Mode B: Star graph (one entity with many relations)
fn mode_b_star_graph(n: usize) {
    let start = Instant::now();
    let mut space = MorkSpace::new();

    // Central entity
    space.add(Atom::expr([Atom::sym("entity"), Atom::sym("central")]));

    // Many relations from central entity
    for i in 0..n {
        space.add(Atom::expr([Atom::sym("relates_to"), Atom::sym("central"), Atom::sym(&format!("target_{}", i))]));
    }

    let load_time = start.elapsed();

    // Query: all relations from central
    let q_start = Instant::now();
    let q = Atom::expr([Atom::sym("relates_to"), Atom::sym("central"), Atom::var("x")]);
    let results = space.query(&q).len();
    let q_time = q_start.elapsed();

    println!("Star graph (n={:>8}): load={:>6}ms, query={:>6}µs, results={}",
             n, load_time.as_millis(), q_time.as_micros(), results);
}

/// Mode B: Dense graph (many entities with shared relations)
fn mode_b_dense_graph(entities: usize, relations_per_entity: usize) {
    let start = Instant::now();
    let mut space = MorkSpace::new();

    // Create entities
    for i in 0..entities {
        space.add(Atom::expr([Atom::sym("entity"), Atom::sym(&format!("e_{}", i))]));
    }

    // Create dense relations (each entity relates to many others)
    for i in 0..entities {
        for j in 0..relations_per_entity {
            let target = (i + j + 1) % entities;
            space.add(Atom::expr([Atom::sym("relates"), Atom::sym(&format!("e_{}", i)), Atom::sym(&format!("e_{}", target))]));
        }
    }

    let load_time = start.elapsed();

    // Query: all relations from entity_0
    let q_start = Instant::now();
    let q = Atom::expr([Atom::sym("relates"), Atom::sym("e_0"), Atom::var("x")]);
    let results = space.query(&q).len();
    let q_time = q_start.elapsed();

    println!("Dense graph ({},{}): load={:>6}ms, query={:>6}µs, results={}",
             entities, relations_per_entity, load_time.as_millis(), q_time.as_micros(), results);
}

/// Mode B: Deep chain (A -> B -> C -> ... -> N)
fn mode_b_deep_chain(depth: usize) {
    let start = Instant::now();
    let mut space = MorkSpace::new();

    // Create chain
    for i in 0..depth {
        space.add(Atom::expr([Atom::sym("next"), Atom::sym(&format!("n_{}", i)), Atom::sym(&format!("n_{}", i+1))]));
    }

    let load_time = start.elapsed();

    // Query: direct successor
    let q_start = Instant::now();
    let q = Atom::expr([Atom::sym("next"), Atom::sym("n_0"), Atom::var("x")]);
    let results = space.query(&q).len();
    let q_time = q_start.elapsed();

    // Query: multi-hop (2 hops)
    let q2_start = Instant::now();
    let q2 = Atom::expr([
        Atom::sym(","),
        Atom::expr([Atom::sym("next"), Atom::sym("n_0"), Atom::var("mid")]),
        Atom::expr([Atom::sym("next"), Atom::var("mid"), Atom::var("x")]),
    ]);
    let q2_results = space.query(&q2).len();
    let q2_time = q2_start.elapsed();

    println!("Deep chain (n={:>8}): load={:>6}ms, hop1={:>6}µs ({}), hop2={:>6}µs ({})",
             depth, load_time.as_millis(), q_time.as_micros(), results, q2_time.as_micros(), q2_results);
}

/// Mode B: Human exploration workload
fn mode_b_human_exploration(n: usize) {
    let start = Instant::now();
    let mut space = MorkSpace::new();

    // Create a knowledge graph similar to what a user would explore
    // Companies with relationships
    for i in 0..n/10 {
        let company = format!("company_{}", i);
        let person = format!("person_{}", i);
        let industry = if i % 3 == 0 { "Tech" } else if i % 3 == 1 { "Automotive" } else { "Finance" };

        space.add(Atom::expr([Atom::sym("isa"), Atom::sym(&company), Atom::sym("Company")]));
        space.add(Atom::expr([Atom::sym("isa"), Atom::sym(&person), Atom::sym("Person")]));
        space.add(Atom::expr([Atom::sym("founded_by"), Atom::sym(&company), Atom::sym(&person)]));
        space.add(Atom::expr([Atom::sym("industry"), Atom::sym(&company), Atom::sym(industry)]));
        space.add(Atom::expr([Atom::sym("source"), Atom::sym(&company), Atom::sym("\"Wikipedia\"")]));
        space.add(Atom::expr([Atom::sym("confidence"), Atom::sym(&company), Atom::sym("0.95")]));

        // Cross-references (person worked at other companies)
        if i > 0 {
            let prev_company = format!("company_{}", i - 1);
            space.add(Atom::expr([Atom::sym("worked_at"), Atom::sym(&person), Atom::sym(&prev_company)]));
        }
    }

    let load_time = start.elapsed();
    let atom_count = space.atom_count().unwrap_or(0);

    // Simulate user exploration: "Show me Tesla and related entities"
    // Cold start: first query
    let cold_start = Instant::now();
    let q1 = Atom::expr([Atom::sym("founded_by"), Atom::sym("company_0"), Atom::var("person")]);
    let _ = space.query(&q1);
    let cold_time = cold_start.elapsed();

    // Warm query: follow relation
    let warm_start = Instant::now();
    let q2 = Atom::expr([Atom::sym("industry"), Atom::sym("company_0"), Atom::var("ind")]);
    let _ = space.query(&q2);
    let warm_time = warm_start.elapsed();

    // Multi-hop: person -> worked_at -> company -> industry
    let multi_start = Instant::now();
    let q3 = Atom::expr([
        Atom::sym(","),
        Atom::expr([Atom::sym("founded_by"), Atom::sym("company_0"), Atom::var("person")]),
        Atom::expr([Atom::sym("worked_at"), Atom::var("person"), Atom::var("other_company")]),
    ]);
    let q3_results = space.query(&q3).len();
    let multi_time = multi_start.elapsed();

    // Filter by metadata
    let filter_start = Instant::now();
    let q4 = Atom::expr([
        Atom::sym(","),
        Atom::expr([Atom::sym("company"), Atom::var("x")]),
        Atom::expr([Atom::sym("source"), Atom::var("x"), Atom::sym("\"Wikipedia\"")]),
        Atom::expr([Atom::sym("confidence"), Atom::var("x"), Atom::sym("0.95")]),
    ]);
    let _ = space.query(&q4);
    let filter_time = filter_start.elapsed();

    println!("Human exploration (n={:>8}): load={:>6}ms, atoms={}", n, load_time.as_millis(), atom_count);
    println!("  Cold start:     {:>6}µs", cold_time.as_micros());
    println!("  Warm query:     {:>6}µs", warm_time.as_micros());
    println!("  Multi-hop:      {:>6}µs ({} results)", multi_time.as_micros(), q3_results);
    println!("  Filter+metadata:{:>6}µs", filter_time.as_micros());
}
