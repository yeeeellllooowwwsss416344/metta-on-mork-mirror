use hyperon_atom::Atom;
use hyperon_space::{Space, SpaceMut};
use metta_on_mork::MorkSpace;

fn main() {
    println!("=== Minimal MORK Query Test ===\n");

    let mut space = MorkSpace::new();

    // Insert facts
    space.add(Atom::expr([Atom::sym("parent"), Atom::sym("Tom"), Atom::sym("Bob")]));
    space.add(Atom::expr([Atom::sym("parent"), Atom::sym("Bob"), Atom::sym("Alice")]));
    space.add(Atom::expr([Atom::sym("parent"), Atom::sym("Tom"), Atom::sym("Charlie")]));
    space.add(Atom::expr([Atom::sym("isa"), Atom::sym("Tom"), Atom::sym("Person")]));
    space.add(Atom::expr([Atom::sym("isa"), Atom::sym("Bob"), Atom::sym("Person")]));
    space.add(Atom::expr([Atom::sym("isa"), Atom::sym("Alice"), Atom::sym("Person")]));
    space.add(Atom::expr([Atom::sym("isa"), Atom::sym("Charlie"), Atom::sym("Person")]));

    println!("Loaded {:?} atoms", space.atom_count());

    // Query 1: who is Tom's child?
    println!("\n--- Query 1: Tom's children ---");
    let q1 = Atom::expr([Atom::sym("parent"), Atom::sym("Tom"), Atom::var("child")]);
    let results1 = space.query(&q1);
    println!("Results: {:?}", results1);

    // Query 2: who are Bob's parents?
    println!("\n--- Query 2: Bob's parents ---");
    let q2 = Atom::expr([Atom::sym("parent"), Atom::var("parent"), Atom::sym("Bob")]);
    let results2 = space.query(&q2);
    println!("Results: {:?}", results2);

    // Query 3: all persons
    println!("\n--- Query 3: All persons ---");
    let q3 = Atom::expr([Atom::sym("isa"), Atom::var("x"), Atom::sym("Person")]);
    let results3 = space.query(&q3);
    println!("Results: {:?}", results3);

    // Query 4: conjunction - find children of Tom who are persons
    println!("\n--- Query 4: Tom's children who are persons ---");
    let q4 = Atom::expr([
        Atom::sym(","),
        Atom::expr([Atom::sym("parent"), Atom::sym("Tom"), Atom::var("child")]),
        Atom::expr([Atom::sym("isa"), Atom::var("child"), Atom::sym("Person")]),
    ]);
    let results4 = space.query(&q4);
    println!("Results: {:?}", results4);

    println!("\n=== All queries completed successfully ===");
}
