/// GroundingSpace diagnostic spike: does Space::query() show the same
/// materialization bottleneck as MorkSpace?

use hyperon_atom::Atom;
use hyperon::space::grounding::GroundingSpace;
use hyperon_space::{Space, SpaceMut};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(1000);

    let load_start = Instant::now();
    let mut space = GroundingSpace::new();

    let companies: Vec<String> = (0..n / 10).map(|i| format!("company_{}", i)).collect();
    let persons: Vec<String> = (0..n / 10).map(|i| format!("person_{}", i)).collect();
    let vehicles: Vec<String> = (0..n / 10).map(|i| format!("vehicle_{}", i)).collect();

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

    let load_time = load_start.elapsed();

    // Q6: Cartesian product of Automotive companies
    let q6 = Atom::expr([
        Atom::sym(","),
        Atom::expr([Atom::sym("industry"), Atom::var("c1"), Atom::sym("Automotive")]),
        Atom::expr([Atom::sym("industry"), Atom::var("c2"), Atom::sym("Automotive")]),
    ]);

    let automotive_count = companies.len();
    let expected = automotive_count * automotive_count;

    let q6_start = Instant::now();
    let results = space.query(&q6);
    let q6_time = q6_start.elapsed();
    let q6_count = results.len();

    println!("GROUNDING N={:>8}: load={:>6}ms", n, load_time.as_millis());
    println!("  Automotive companies: {}", automotive_count);
    println!("  Q6 expected: {} ({}²)", expected, automotive_count);
    println!("  Q6 results:  {}, time: {:>6}ms", q6_count, q6_time.as_millis());
    println!("  Q6 match: {}", if q6_count == expected { "PASS" } else { "MISMATCH" });

    // Print sample bindings
    if q6_count > 0 {
        println!("  --- Sample bindings (first 3):");
        for (i, binding) in results.iter().take(3).enumerate() {
            let vars: Vec<String> = binding.iter().map(|(v, a)| format!("{}={}", v, a)).collect();
            println!("    [{}] {}", i + 1, vars.join(", "));
        }
        println!("  ---");
    }
}
