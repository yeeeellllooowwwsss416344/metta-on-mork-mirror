/// F1 Vertical Slice: WordNet subset → MORK → queries
/// 
/// Runs the same workload as test_f1_vertical_slice.py through MorkSpace
/// to verify MORK produces equivalent results to GroundingSpace.

use hyperon_atom::{Atom, VariableAtom};
use hyperon_space::{Space, SpaceMut};
use metta_on_mork::MorkSpace;

fn main() {
    println!("=== F1 Vertical Slice: MORK Workload Test ===\n");

    let mut space = MorkSpace::new();

    // Step 1: Load atoms (equivalent to converter + materializer output)
    // isa hierarchy (raw edges)
    space.add(Atom::expr([Atom::sym("isa"), Atom::sym("dog"), Atom::sym("animal")]));
    space.add(Atom::expr([Atom::sym("isa"), Atom::sym("cat"), Atom::sym("animal")]));
    space.add(Atom::expr([Atom::sym("isa"), Atom::sym("animal"), Atom::sym("living_thing")]));
    space.add(Atom::expr([Atom::sym("isa"), Atom::sym("plant"), Atom::sym("living_thing")]));
    space.add(Atom::expr([Atom::sym("isa"), Atom::sym("living_thing"), Atom::sym("entity")]));

    // Transitive isa (materializer output: materialized_isa_atoms)
    space.add(Atom::expr([Atom::sym("isa"), Atom::sym("dog"), Atom::sym("living_thing")]));
    space.add(Atom::expr([Atom::sym("isa"), Atom::sym("cat"), Atom::sym("living_thing")]));
    space.add(Atom::expr([Atom::sym("isa"), Atom::sym("dog"), Atom::sym("entity")]));
    space.add(Atom::expr([Atom::sym("isa"), Atom::sym("cat"), Atom::sym("entity")]));
    space.add(Atom::expr([Atom::sym("isa"), Atom::sym("animal"), Atom::sym("entity")]));
    space.add(Atom::expr([Atom::sym("isa"), Atom::sym("plant"), Atom::sym("entity")]));

    // source_of provenance
    space.add(Atom::expr([Atom::sym("source_of"), Atom::sym("dog"), Atom::sym("\"Princeton WordNet 3.1\"")]));
    space.add(Atom::expr([Atom::sym("source_of"), Atom::sym("cat"), Atom::sym("\"Princeton WordNet 3.1\"")]));
    space.add(Atom::expr([Atom::sym("source_of"), Atom::sym("animal"), Atom::sym("\"Princeton WordNet 3.1\"")]));
    space.add(Atom::expr([Atom::sym("source_of"), Atom::sym("plant"), Atom::sym("\"Princeton WordNet 3.1\"")]));
    space.add(Atom::expr([Atom::sym("source_of"), Atom::sym("living_thing"), Atom::sym("\"Princeton WordNet 3.1\"")]));
    space.add(Atom::expr([Atom::sym("source_of"), Atom::sym("entity"), Atom::sym("\"Princeton WordNet 3.1\"")]));

    // Bridge rule (materializer)
    space.add(Atom::expr([
        Atom::sym("="),
        Atom::expr([Atom::sym("isa"), Atom::var("x"), Atom::var("y")]),
        Atom::expr([Atom::sym("isa"), Atom::var("x"), Atom::var("y")]),
    ]));

    println!("Loaded {:?} atoms\n", space.atom_count());

    // Query 1: taxonomy direct (children of animal)
    println!("--- Query 1: taxonomy direct ---");
    let q1 = Atom::expr([Atom::sym("isa"), Atom::var("x"), Atom::sym("animal")]);
    let results1 = space.query(&q1);
    let mut q1_values: Vec<String> = results1.iter()
        .filter_map(|b| b.resolve(&VariableAtom::new("x")))
        .map(|a| format!("{:?}", a))
        .collect();
    q1_values.sort();
    println!("Results: {:?}", q1_values);
    let q1_pass = q1_values.contains(&"dog".to_string()) 
        && q1_values.contains(&"cat".to_string());
    println!("PASS: {}\n", q1_pass);

    // Query 2: taxonomy transitive (ancestors of dog)
    println!("--- Query 2: taxonomy transitive ---");
    let q2 = Atom::expr([Atom::sym("isa"), Atom::sym("dog"), Atom::var("x")]);
    let results2 = space.query(&q2);
    let mut q2_values: Vec<String> = results2.iter()
        .filter_map(|b| b.resolve(&VariableAtom::new("x")))
        .map(|a| format!("{:?}", a))
        .collect();
    q2_values.sort();
    println!("Results: {:?}", q2_values);
    let q2_pass = q2_values.contains(&"living_thing".to_string());
    println!("PASS: {}\n", q2_pass);

    // Query 3: provenance
    println!("--- Query 3: provenance ---");
    let q3 = Atom::expr([Atom::sym("source_of"), Atom::sym("dog"), Atom::var("x")]);
    let results3 = space.query(&q3);
    let q3_values: Vec<String> = results3.iter()
        .filter_map(|b| b.resolve(&VariableAtom::new("x")))
        .map(|a| format!("{:?}", a))
        .collect();
    println!("Results: {:?}", q3_values);
    let q3_pass = q3_values.contains(&"\"Princeton WordNet 3.1\"".to_string());
    println!("PASS: {}\n", q3_pass);

    // Query 4: conjunction (dog and cat are animals)
    println!("--- Query 4: conjunction ---");
    let q4 = Atom::expr([
        Atom::sym(","),
        Atom::expr([Atom::sym("isa"), Atom::sym("dog"), Atom::var("x")]),
        Atom::expr([Atom::sym("isa"), Atom::sym("cat"), Atom::var("x")]),
    ]);
    let results4 = space.query(&q4);
    let q4_count = results4.len();
    println!("Results: {} bindings", q4_count);
    let q4_pass = q4_count > 0;
    println!("PASS: {}\n", q4_pass);

    // Summary
    let queries_passed = [q1_pass, q2_pass, q3_pass, q4_pass].iter().filter(|&&x| x).count();
    println!("=== Summary: {}/4 queries passed ===", queries_passed);
    println!("  Atom count: {:?}", space.atom_count());
    println!("  Query 1 (taxonomy direct): {}", if q1_pass { "PASS" } else { "FAIL" });
    println!("  Query 2 (taxonomy transitive): {}", if q2_pass { "PASS" } else { "FAIL" });
    println!("  Query 3 (provenance): {}", if q3_pass { "PASS" } else { "FAIL" });
    println!("  Query 4 (conjunction): {}", if q4_pass { "PASS" } else { "FAIL" });

    assert_eq!(queries_passed, 4, "All 4 queries must pass");
    println!("\n=== All queries passed ===");
}
