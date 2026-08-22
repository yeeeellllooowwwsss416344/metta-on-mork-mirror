// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 MesTTo
//! MORK kernel as an in-process Hyperon atomspace backend (MeTTa-On-Mork).
//!
//! [`MorkSpace`] implements Hyperon's [`Space`]/[`SpaceMut`] over the optimized MORK
//! kernel (a PathMap trie plus the worst-case-optimal-join matcher), so a Hyperon
//! atomspace gains MORK's scale and speed. Motivated by hyperon-experimental #1076
//! (the GroundingSpace trie panics on the first query past ~2k atoms).
//!
//! The codec is **byte-level**: a Hyperon `Atom` is walked directly into MORK's
//! expression-zipper byte encoding (`Arity`/`SymbolSize`/`NewVar`/`VarRef`) and
//! inserted into the trie; a query encodes its pattern, calls `query_multi`, and
//! decodes each bound sub-expression's bytes straight back into an `Atom`. No text
//! round-trip, no per-query parsing. (Build uses MORK's default features, where
//! symbols are stored as raw bytes rather than interned tokens.)

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use hyperon_atom::matcher::{match_atoms, Bindings, BindingsSet};
use hyperon_atom::{next_variable_id, Atom, Grounded, GroundedAtom, VariableAtom};
use hyperon_common::FlexRef;
use hyperon_space::{Space, SpaceCommon, SpaceMut, SpaceVisitor};

use mork::__mork_expr::{byte_item, item_byte, Expr, Tag};
use mork::space::Space as MorkKernel;
use pathmap::PathMap;

use crate::william::{decode_pattern, WeightedPathIndex};

/// Priority ordering for evaluation control (Hyperon #448), grabbed from MeTTaTron.
pub mod priority;

/// WILLIAM's compression-gain index and pattern report, carried by this crate
/// (the upstream kernel's weighted_paths sidecar stops at weight bookkeeping).
pub mod william;

/// The argument-position (column) index: selective non-leading-bound queries
/// seek a maintained permuted-key trie instead of scanning the relation.
pub mod argindex;

/// MORK's 6-bit `SymbolSize`/`Arity` fields cap symbol length and arity at 63.
const MAX_FIELD: usize = 63;

/// Tabling caps: at most this many distinct query shapes stay cached per space
/// (insertion-capped; a stale entry re-fills in place past the cap)...
const QUERY_CACHE_MAX_ENTRIES: usize = 256;
/// ...and a result set larger than this skips the cache, bounding memory.
const QUERY_CACHE_MAX_ROWS: usize = 4096;
/// Auto-tabling admission: only a fill whose matcher run cost at least this
/// much is tabled. Replay saves exactly the matcher's cost, so a cheap query
/// (a warm point lookup runs in a few microseconds) is never worth a cache
/// entry; the shapes that qualify are the scan-class ones replay turns
/// O(N) -> O(answers). This keeps the cache's memory proportional to queries
/// that actually pay, not to query traffic.
/// Calibrated for release; debug builds run the matcher an order of magnitude
/// slower, so their threshold scales with them.
const QUERY_TABLE_MIN_COST: std::time::Duration = if cfg!(debug_assertions) {
    // Wide margins for the debug-build tests: a 1-atom fill stays far below
    // even under load spikes, a 30k-store scan stays far above.
    std::time::Duration::from_millis(5)
} else {
    std::time::Duration::from_micros(50)
};

/// The auto-tabling admission predicate: table only fills that cost enough to
/// pay for their memory, and never oversized result sets.
fn worth_tabling(fill_cost: std::time::Duration, rows: usize) -> bool {
    fill_cost >= QUERY_TABLE_MIN_COST && rows <= QUERY_CACHE_MAX_ROWS
}

/// Reserved first byte of a grounded atom's symbol encoding. No real MeTTa symbol
/// starts with NUL, so it cleanly separates a grounded `Atom` from a plain symbol in
/// the trie. The bytes after it are the grounded atom's display string, the key into
/// [`GroundedRegistry`]. Used for content-addressable (immutable) grounded atoms:
/// numbers, bools, strings, and operation symbols, whose display is their identity.
const GROUNDED_MARK: u8 = 0x00;

/// Reserved first byte for a *mutable* grounded atom (`Grounded::is_mutable`), e.g. a
/// `State` cell. Its display is unstable, so it is addressed by a stable per-instance
/// id (the 8 bytes after this marker) rather than by content; the live `Atom` is held
/// in [`GroundedRegistry::by_id`]. Like `0x00`, no real symbol starts with `0x01`.
const GROUNDED_REF_MARK: u8 = 0x01;

/// Returns true if `atom` is a *mutable* grounded atom (`Grounded::is_mutable`), e.g. a
/// `State` cell, whose display is unstable so it cannot be content-addressed.
fn is_mutable_grounded(atom: &Atom) -> bool {
    matches!(atom, Atom::Grounded(g) if g.as_grounded().is_mutable())
}

/// Length of the encoded pattern's ground prefix: its bytes up to the first
/// variable item, scanning item-by-item so a `$` inside a symbol payload is
/// never mistaken for a variable tag.
fn ground_prefix_len(bytes: &[u8]) -> usize {
    let mut pos = 0usize;
    while pos < bytes.len() {
        match mork::__mork_expr::maybe_byte_item(bytes[pos]) {
            Ok(Tag::NewVar) | Ok(Tag::VarRef(_)) | Err(_) => return pos,
            Ok(Tag::Arity(_)) => pos += 1,
            Ok(Tag::SymbolSize(size)) => pos += 1 + size as usize,
        }
    }
    pos
}

/// Whether an encoded fact span holds only ground bytes (no NewVar/VarRef tag).
/// A ground fact has no helper variables to leak, so its raw-row replay is
/// exactly the narrowed result and the per-row Hyperon rematch can be skipped;
/// this keeps the byte-seek constants (column-index seeks, warm point lookups)
/// on var-free data, which is the latch's whole class.
fn span_is_ground(bytes: &[u8]) -> bool {
    fn walk(bytes: &[u8], pos: &mut usize) -> Option<bool> {
        let tag = byte_item(*bytes.get(*pos)?);
        *pos += 1;
        match tag {
            Tag::SymbolSize(size) => {
                *pos = (*pos).checked_add(size as usize)?;
                bytes.get(..*pos)?;
                Some(true)
            }
            Tag::Arity(k) => {
                for _ in 0..k {
                    if !walk(bytes, pos)? {
                        return Some(false);
                    }
                }
                Some(true)
            }
            Tag::NewVar | Tag::VarRef(_) => Some(false),
        }
    }
    let mut pos = 0usize;
    walk(bytes, &mut pos) == Some(true)
}

fn skip_encoded_atom(bytes: &[u8], pos: &mut usize) -> Option<()> {
    let tag = byte_item(*bytes.get(*pos)?);
    *pos += 1;
    match tag {
        Tag::SymbolSize(size) => {
            *pos = (*pos).checked_add(size as usize)?;
            bytes.get(..*pos)?;
        }
        Tag::Arity(k) => {
            for _ in 0..k {
                skip_encoded_atom(bytes, pos)?;
            }
        }
        Tag::NewVar | Tag::VarRef(_) => {}
    }
    Some(())
}

fn push_specificity_key(pattern: &[u8], p: &mut usize, fact: &[u8], f: &mut usize, out: &mut Vec<u8>) -> Option<()> {
    let p_start = *p;
    let f_start = *f;
    let p_tag = byte_item(*pattern.get(*p)?);
    let f_tag = byte_item(*fact.get(*f)?);
    if matches!(p_tag, Tag::NewVar | Tag::VarRef(_)) {
        out.push(if matches!(f_tag, Tag::NewVar | Tag::VarRef(_)) {
            0
        } else {
            1
        });
        *p += 1;
        skip_encoded_atom(fact, f)?;
        return Some(());
    }
    if matches!(f_tag, Tag::NewVar | Tag::VarRef(_)) {
        out.push(1);
        skip_encoded_atom(pattern, p)?;
        *f += 1;
        return Some(());
    }
    out.push(0);
    *p += 1;
    *f += 1;
    match (p_tag, f_tag) {
        (Tag::SymbolSize(p_size), Tag::SymbolSize(f_size)) => {
            *p = (*p).checked_add(p_size as usize)?;
            *f = (*f).checked_add(f_size as usize)?;
            pattern.get(p_start..*p)?;
            fact.get(f_start..*f)?;
        }
        (Tag::Arity(p_arity), Tag::Arity(f_arity)) if p_arity == f_arity => {
            for _ in 0..p_arity {
                push_specificity_key(pattern, p, fact, f, out)?;
            }
        }
        _ => {
            *p = p_start;
            *f = f_start;
            skip_encoded_atom(pattern, p)?;
            skip_encoded_atom(fact, f)?;
        }
    }
    Some(())
}

fn specificity_key(pattern: &[u8], fact: &[u8]) -> Vec<u8> {
    let mut p = 0usize;
    let mut f = 0usize;
    let mut key = Vec::new();
    let _ = push_specificity_key(pattern, &mut p, fact, &mut f, &mut key);
    key
}

/// Whether `atom` holds a mutable grounded value anywhere in its tree: such
/// atoms need the space's identity registry at encode time.
fn contains_mutable_grounded(atom: &Atom) -> bool {
    match atom {
        Atom::Grounded(g) => g.as_grounded().is_mutable(),
        Atom::Expression(e) => e.children().iter().any(contains_mutable_grounded),
        _ => false,
    }
}

/// Side table that makes grounded atoms round-trip losslessly through the byte trie.
///
/// MORK's encoding has no grounded type, so a Hyperon `Grounded` atom is kept here and
/// referenced from the trie by a marker symbol. Two storage disciplines, chosen by
/// [`Grounded::is_mutable`]:
///
/// - **Immutable** atoms (numbers, bools, strings, operation symbols like `<=`): keyed
///   by display string in `by_display`. Encoding is deterministic from the display, so a
///   query pattern encodes to the same bytes as stored data without touching the table,
///   and two atoms that print the same are the same. This is content addressing.
/// - **Mutable** atoms (a `State` cell): keyed by a fresh per-instance id in `by_id`,
///   because the display goes stale on `change-state!` and two cells with equal values
///   are still distinct. The stored `Atom` is an `Rc`-sharing clone, so the registered
///   handle and the live cell mutate together (the handle-table model of the minimal
///   interpreter's `World.store`). Matching a mutable atom is by *live* value, handled
///   as a post-filter in [`query_btm`], not by these bytes.
///
/// Snapshots and sharded shards carry no registry, so they decode grounded atoms as bare
/// symbols (and do not support mutable-cell identity).
#[derive(Default, Clone)]
struct GroundedRegistry {
    by_display: HashMap<String, Atom>,
    by_id: HashMap<u64, Atom>,
    next_id: u64,
}

impl GroundedRegistry {
    /// Records every *immutable* grounded atom in `atom` (recursing into expressions) by
    /// display so it can be reconstructed on decode. Mutable atoms are skipped here; they
    /// are interned by identity in [`intern_mutable`](Self::intern_mutable) during encode.
    /// First instance per display string wins.
    fn register(&mut self, atom: &Atom) {
        match atom {
            Atom::Grounded(g) if !g.as_grounded().is_mutable() => {
                self.by_display
                    .entry(g.to_string())
                    .or_insert_with(|| atom.clone());
            }
            Atom::Expression(e) => {
                for c in e.children() {
                    self.register(c);
                }
            }
            _ => {}
        }
    }

    /// Interns a mutable grounded atom by a fresh id, storing an `Rc`-sharing clone so the
    /// registered handle tracks the live cell. Returns the id to embed in the trie bytes.
    fn intern_mutable(&mut self, atom: &Atom) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.by_id.insert(id, atom.clone());
        id
    }

    fn get(&self, display: &str) -> Option<Atom> {
        self.by_display.get(display).cloned()
    }

    fn get_by_id(&self, id: u64) -> Option<Atom> {
        self.by_id.get(&id).cloned()
    }
}

#[derive(Default)]
struct EvaluatedSpans {
    spans: Vec<Vec<u8>>,
}

impl EvaluatedSpans {
    fn insert(&mut self, span: &[u8]) {
        self.spans.push(span.to_vec());
    }

    fn contains(&self, span: &[u8]) -> bool {
        self.spans.iter().any(|recorded| recorded == span)
    }
}

struct WrappedPattern {
    bytes: Vec<u8>,
    vars: Vec<VariableAtom>,
    refs: Vec<(usize, Atom)>,
    evaluated: EvaluatedSpans,
}

/// Decode-time state: the namespace these bytes belong to (so variables are named by
/// `(namespace, index)` and a data variable can't collide with a pattern variable), a
/// running counter for first-occurrence variables, the optional grounded registry, and
/// the query's own variables. Namespace 0 is the query pattern, so a namespace-0 variable
/// decodes back to the caller's actual `VariableAtom` (preserving `$n` through the byte
/// round-trip) rather than a fresh name; without this, evaluating `(Add $n (S Z))` returns
/// an alpha-equivalent `(S $v0_0)` that the script's `assertEqual` against `(S $n)` rejects.
struct DecodeCtx<'a> {
    ns: u8,
    var_counter: usize,
    grounded: Option<&'a GroundedRegistry>,
    query_vars: &'a [VariableAtom],
    evaluated: Option<&'a EvaluatedSpans>,
    /// Globally unique id stamped on every data variable decoded for one query result,
    /// so two results (or two separate query calls, e.g. recursion levels) never produce
    /// the same `VariableAtom` and alias when the interpreter threads their bindings.
    result_id: usize,
}

/// A Hyperon atomspace backed by the MORK kernel.
///
/// The byte-level matcher (`query_multi`) takes the trie by shared reference, so
/// `query`/`visit` read the kernel through `&self` and mutation goes through
/// `&mut self`. The full space is still not `Sync` because Hyperon's `SpaceCommon`,
/// MORK/PathMap internals, and grounded atoms carry non-`Sync` state. Use
/// [`MorkSnapshot`] when many threads need to query one immutable view.
pub struct MorkSpace {
    kernel: MorkKernel,
    common: SpaceCommon,
    grounded: GroundedRegistry,
    rejected_atoms: usize,
    /// Argument-position indexes, keyed by (functor prefix, arg count, indexed
    /// position), each carrying the mutation generation it was built at. A
    /// query bound only on non-leading positions seeks the index instead of
    /// scanning the relation; a stale entry (older generation) is rebuilt on
    /// use. Interior mutability because `query` is `&self`; an RwLock so the
    /// space itself stays `Send + Sync`-capable.
    column_index_cache: std::sync::RwLock<HashMap<(Vec<u8>, usize, usize), (PathMap<()>, u64)>>,
    /// Bumped by every mutation (`add`, `remove`, `add_sexpr_text`, `step`):
    /// the O(1) staleness authority for the column indexes and the query cache.
    mutation_gen: u64,
    /// Tabled queries: the kernel matcher's raw rows per encoded pattern, keyed
    /// by the `(, ...)` pattern bytes (alpha-invariant by construction) and the
    /// mutation generation they were captured at. A hit replays rows through
    /// [`rows_to_bindings`], whose per-row fresh ids and live mutable-grounded
    /// filter make a replay indistinguishable from a live match; a repeated
    /// scan-class query is O(answers) instead of O(N) from the second call on.
    query_cache: std::sync::RwLock<HashMap<Vec<u8>, (Vec<RawRow>, u64)>>,
    /// One-way latch: true while every stored atom is variable-free. Byte-level
    /// fast paths (the factorized conjunctive count) match by trie prefix and
    /// cannot see a stored variable unifying with a pattern position (a bare
    /// `$x` fact matches *any* factor, which a prefix seek returns zero for),
    /// so they are admitted only while this holds. Cleared by a variable-bearing
    /// `add`, by `add_sexpr_text` whose text can parse variables, and by `step`
    /// (whose templates may write variables); never re-set, so a ground
    /// relational store keeps its fast paths for its whole life.
    var_free: bool,
}

impl MorkSpace {
    /// Creates an empty space.
    pub fn new() -> Self {
        Self {
            kernel: MorkKernel::new(),
            common: SpaceCommon::default(),
            grounded: GroundedRegistry::default(),
            rejected_atoms: 0,
            column_index_cache: std::sync::RwLock::new(HashMap::new()),
            mutation_gen: 0,
            query_cache: std::sync::RwLock::new(HashMap::new()),
            var_free: true,
        }
    }

    /// Number of atoms currently stored (MORK `PathMap::val_count`).
    pub fn len(&self) -> usize {
        self.kernel.btm.val_count()
    }

    /// Whether the space holds no atoms.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of atoms rejected by `add` because they do not fit MORK's byte encoding.
    ///
    /// The common reasons are a symbol/display string longer than 63 bytes, expression
    /// arity above 63, or more than 64 distinct variables in one atom.
    pub fn rejected_atom_count(&self) -> usize {
        self.rejected_atoms
    }

    /// Bulk-loads atoms from an S-expression source (whitespace-separated atoms)
    /// through MORK's parser. Convenient for large loads.
    pub fn add_sexpr_text(&mut self, text: &str) -> Result<usize, String> {
        if text.as_bytes().contains(&b'$') {
            self.var_free = false;
        }
        self.mutation_gen += 1;
        self.kernel.add_all_sexpr(text.as_bytes())
    }

    /// WILLIAM (whitepaper 5.12): the term-boundary compression-gain index over the
    /// stored atoms. Every whole-subexpression prefix shared by `count >= 2` atoms is
    /// weighted `(count - 1) * len - count * ref_cost` (the bytes factoring it would
    /// save); the index's top-k iterator surfaces the heaviest patterns without a
    /// store scan. `ref_cost` is the byte cost charged per reference to a factored
    /// definition; the fork's validated factoring loop uses 9 (one `SymbolSize` tag
    /// plus an 8-byte reference id payload).
    pub fn compression_gain_index(&self, ref_cost: u64) -> WeightedPathIndex {
        WeightedPathIndex::from_compression_gain_on_boundaries(&self.kernel.btm, ref_cost)
    }

    /// The `k` heaviest non-overlapping compressible subpatterns, rendered as MeTTa
    /// (argument slots the pattern cuts off show as `…`), heaviest first. One
    /// representative per hot chain: a pattern nested inside a chosen heavier one is
    /// suppressed, so the report reads as distinct structures rather than one chain's
    /// prefixes.
    pub fn frequent_subpatterns(&self, k: usize, ref_cost: u64) -> Vec<(String, u64)> {
        self.compression_gain_index(ref_cost)
            .iter_any_topk_maximal(k)
            .map(|entries| {
                entries
                    .into_iter()
                    .map(|(pattern, gain)| (decode_pattern(&pattern), gain))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Runs MORK's MM2 exec engine -- the forward-chaining `(exec <loc> (, <src>)
    /// (, <tpl>))` rules in the space -- for up to `steps` exec steps (each runs a
    /// rule to fixpoint), returning the steps taken. This is the *computation*
    /// engine (CeTTa's `mork:step!`). Load facts and `exec` rules with
    /// `add`/`add_sexpr_text` first, then `step`, then `query` the results.
    ///
    /// Complexity opt-ins (cargo features, each byte-identical by the kernel's
    /// differentials): `semi-naive` re-derives only each round's delta instead of
    /// the whole space per round; `leapfrog` routes flat conjunctive rule bodies
    /// through the worst-case-optimal leapfrog join; `factorized-aggregate` runs
    /// COUNT/SUM/MIN/MAX/AND sinks without enumerating the join.
    pub fn step(&mut self, steps: usize) -> usize {
        if steps > 0 {
            self.var_free = false;
            self.mutation_gen += 1;
        }
        self.kernel.metta_calculus(steps)
    }

    fn query_inner(&self, query: &Atom) -> BindingsSet {
        if self.var_free {
            if let Some(set) = self.indexed_query(query) {
                return set;
            }
        }
        // The empty conjunction `(,)` is a fold over zero sub-queries.
        if conjuncts(query).is_some_and(|qs| qs.is_empty()) {
            return BindingsSet::single();
        }
        let Some(pattern) = wrap_pattern(query) else {
            return BindingsSet::empty();
        };
        let mut reg = self.grounded.clone();
        reg.register(query);
        // Tabling: replay this pattern's raw rows while the space is unchanged.
        // Decode runs per hit (fresh result ids) and the mutable-grounded live
        // filter reruns inside rows_to_bindings, so a replayed answer is
        // indistinguishable from a live match.
        let gen = self.mutation_gen;
        {
            let cache = self.query_cache.read().unwrap();
            if let Some((rows, g)) = cache.get(&pattern.bytes) {
                if *g == gen {
                    return rows_to_bindings(
                        rows,
                        &pattern.vars,
                        &pattern.refs,
                        &reg,
                        &pattern.evaluated,
                        conjuncts(query).is_none().then_some(query),
                    );
                }
            }
        }
        let fill_started = std::time::Instant::now();
        let rows = query_btm_rows(&self.kernel.btm, &pattern.bytes);
        let fill_cost = fill_started.elapsed();
        if worth_tabling(fill_cost, rows.len()) {
            let mut cache = self.query_cache.write().unwrap();
            if cache.len() < QUERY_CACHE_MAX_ENTRIES || cache.contains_key(&pattern.bytes) {
                cache.insert(pattern.bytes.clone(), (rows.clone(), gen));
            }
        }
        rows_to_bindings(
            &rows,
            &pattern.vars,
            &pattern.refs,
            &reg,
            &pattern.evaluated,
            conjuncts(query).is_none().then_some(query),
        )
    }

    /// Diagnostic spike: raw callback count without any decode or materialization.
    /// Returns the number of kernel callbacks (each result the kernel matched).
    pub fn query_count_raw(&self, query: &Atom) -> usize {
        if conjuncts(query).is_some_and(|qs| qs.is_empty()) {
            return 1;
        }
        let Some(pattern) = wrap_pattern(query) else {
            return 0;
        };
        query_btm_rows_count_raw(&self.kernel.btm, &pattern.bytes)
    }

    /// Diagnostic spike: decoded callback count without Vec accumulation.
    /// Decodes each binding inline and drops it. Returns
    /// (total_callbacks, decode_ok, decode_errors).
    pub fn query_count_decoded(&self, query: &Atom) -> (usize, usize, usize) {
        if conjuncts(query).is_some_and(|qs| qs.is_empty()) {
            return (1, 1, 0);
        }
        let Some(pattern) = wrap_pattern(query) else {
            return (0, 0, 0);
        };
        let mut reg = self.grounded.clone();
        reg.register(query);
        query_btm_rows_count_decoded(
            &self.kernel.btm,
            &pattern.bytes,
            &pattern.vars,
            &pattern.refs,
            &reg,
            &pattern.evaluated,
            conjuncts(query).is_none().then_some(query),
        )
    }

    /// Incrementally maintains the fresh column indexes across one mutation:
    /// each cached index of the mutated fact's relation absorbs the fact as
    /// one O(1) permuted-key insert or remove, instead of being invalidated
    /// and paying the O(N) rebuild on the next selective query (the
    /// incremental-delta shape of alloy-mork's fac4). Indexes of other
    /// relations are untouched by the mutation and stay valid; an entry that
    /// was already stale stays stale and rebuilds on next use.
    fn update_column_indexes(&mut self, bytes: &[u8], add: bool) {
        let prev_gen = self.mutation_gen;
        self.mutation_gen += 1;
        let gen = self.mutation_gen;
        let mut cache = self.column_index_cache.write().unwrap();
        for ((prefix, ncols, pos), (index, entry_gen)) in cache.iter_mut() {
            if *entry_gen != prev_gen {
                continue;
            }
            if bytes.starts_with(prefix) {
                match argindex::permuted_fact_key(&bytes[prefix.len()..], *ncols, *pos) {
                    Some(key) => {
                        if add {
                            index.insert(&key, ());
                        } else {
                            index.remove(&key);
                        }
                        *entry_gen = gen;
                    }
                    // Unsplittable under this shape: leave the entry stale.
                    None => {}
                }
            } else {
                *entry_gen = gen;
            }
        }
    }

    /// Answers a single-factor query bound only on non-leading argument
    /// positions through the maintained column index (see [`argindex`]): a
    /// prefix seek plus a residual byte filter instead of the matcher's
    /// relation scan. Admitted only on a variable-free store (the latch) and
    /// for the classifier's fragment; `None` falls back to the general
    /// matcher. Agreement with the matcher is sealed by this crate's proptest
    /// differential.
    fn indexed_query(&self, query: &Atom) -> Option<BindingsSet> {
        let (classified, pos, value, vars, refs) = classify_index_route(query)?;
        let ncols = classified.args.len();
        let key = (classified.functor_prefix.clone(), ncols, pos);
        // Seek under the read lock when the index is fresh, so concurrent
        // threads sharing one space seek in parallel; only a miss or a stale
        // entry takes the write lock to (re)build.
        let fresh_hit = {
            let cache = self.column_index_cache.read().unwrap();
            cache.get(&key).and_then(|(index, g)| {
                (*g == self.mutation_gen).then(|| {
                    argindex::arg_index_seek(index, &classified.functor_prefix, ncols, pos, &value)
                })
            })
        };
        let facts = match fresh_hit {
            Some(facts) => facts,
            None => {
                let mut cache = self.column_index_cache.write().unwrap();
                let entry = cache
                    .entry(key)
                    .or_insert_with(|| (PathMap::new(), u64::MAX));
                if entry.1 != self.mutation_gen {
                    entry.0 = argindex::build_arg_index(
                        &self.kernel.btm,
                        &classified.functor_prefix,
                        ncols,
                        pos,
                    );
                    entry.1 = self.mutation_gen;
                }
                argindex::arg_index_seek(&entry.0, &classified.functor_prefix, ncols, pos, &value)
            }
        };

        Some(indexed_facts_to_bindings(
            &facts, &classified, pos, &vars, &refs, self.grounded.clone(), query,
        ))
    }

    /// Runs one multi-pattern -> multi-template rewrite directly: the wiki's
    /// Transform space operation (and the MORK HTTP server's `transform`
    /// command), without installing exec atoms or running the scheduler. One
    /// worst-case-optimal join over the conjunction of `patterns`, one
    /// templated emit per match; a variable shared between patterns is a join
    /// variable, and template variables refer to pattern variables. Returns
    /// `(matches, any_new)`; `None` when something does not encode or a
    /// mutable grounded value appears (byte-level transform cannot honor its
    /// live-value semantics).
    pub fn transform(&mut self, patterns: &[Atom], templates: &[Atom]) -> Option<(usize, bool)> {
        if patterns.is_empty()
            || templates.is_empty()
            || patterns.len() > MAX_FIELD - 1
            || templates.len() > MAX_FIELD - 1
        {
            return None;
        }
        for atom in patterns.iter().chain(templates) {
            if contains_mutable_grounded(atom) {
                return None;
            }
            // Emitted atoms may carry grounded markers; register them so later
            // decodes recover the values.
            self.grounded.register(atom);
        }
        let mut conj_p = vec![Atom::sym(",")];
        conj_p.extend(patterns.iter().cloned());
        let mut conj_t = vec![Atom::sym(",")];
        conj_t.extend(templates.iter().cloned());
        let exec = Atom::expr([
            Atom::sym("exec"),
            Atom::sym("0"),
            Atom::expr(conj_p),
            Atom::expr(conj_t),
        ]);
        // One buffer for the whole exec atom, then in-place offsets to the
        // pattern and template sub-expressions (the kernel's own convention:
        // the template's VarRefs stay relative to the pattern's variables).
        let mut vars = Vec::new();
        let mut bytes = Vec::new();
        let mut refs = Vec::new();
        let mut sink = GroundedSink::Query(&mut refs);
        if encode_atom(&exec, &mut vars, &mut bytes, &mut sink).is_err() || !refs.is_empty() {
            return None;
        }
        let head_len = 1 + argindex::fact_subterm_len(&bytes[1..]);
        let loc_len = argindex::fact_subterm_len(&bytes[head_len..]);
        let pat_off = head_len + loc_len;
        let pat_len = argindex::fact_subterm_len(&bytes[pat_off..]);
        let tpl_off = pat_off + pat_len;
        let pat_expr = Expr {
            ptr: bytes[pat_off..].as_ptr() as *mut u8,
        };
        let tpl_expr = Expr {
            ptr: bytes[tpl_off..].as_ptr() as *mut u8,
        };
        // The kernel inserts its third argument into the read snapshot (MM2
        // rules may match rules). Space-level transform must match stored
        // atoms only, so anchor on an atom that is already present -- a set
        // no-op -- instead of the exec scaffold; an empty space has nothing to
        // match at all.
        let anchor: Vec<u8> = {
            use pathmap::zipper::{ZipperIteration, ZipperMoving};
            let mut rz = self.kernel.btm.read_zipper();
            if !rz.to_next_val() {
                return Some((0, false));
            }
            rz.path().to_vec()
        };
        let anchor_expr = Expr {
            ptr: anchor.as_ptr() as *mut u8,
        };
        let result = self
            .kernel
            .transform_multi_multi_(pat_expr, tpl_expr, anchor_expr);
        // Templates may write variables and everything derived is stale.
        self.var_free = false;
        self.mutation_gen += 1;
        Some(result)
    }

    /// Restriction, the wiki's `x <| y` as a pattern-prefix gate: keeps only
    /// the atoms under `pattern`'s ground prefix -- its encoded bytes up to
    /// the first variable, per Data-in-MORK's "construct values with a ground
    /// prefix, defer variables to the suffix". One O(prefix) descent selects
    /// the whole surviving subtree by structure; nothing is enumerated.
    /// Returns the surviving atom count, or `None` for an unencodable pattern
    /// or one carrying a mutable grounded value.
    pub fn restrict_to_prefix(&mut self, pattern: &Atom) -> Option<usize> {
        use pathmap::zipper::{ZipperInfallibleSubtries, ZipperWriting};
        if contains_mutable_grounded(pattern) {
            return None;
        }
        let mut vars = Vec::new();
        let mut bytes = Vec::new();
        if encode_atom(pattern, &mut vars, &mut bytes, &mut GroundedSink::ValueOnly).is_err() {
            return None;
        }
        let prefix = &bytes[..ground_prefix_len(&bytes)];
        let sub = self.kernel.btm.read_zipper_at_path(prefix).make_map();
        let mut restricted = PathMap::new();
        if sub.val_count() > 0 || self.kernel.btm.get_val_at(prefix).is_some() {
            restricted.write_zipper_at_path(prefix).graft_map(sub);
            if self.kernel.btm.get_val_at(prefix).is_some() {
                restricted.insert(prefix, ());
            }
        }
        self.kernel.btm = restricted;
        self.mutation_gen += 1;
        Some(self.kernel.btm.val_count())
    }

    /// Serializes the space's paths to a compact binary file (PathMap's paths
    /// serialization; the wiki's write-the-space-to-disk lane). Grounded
    /// registries are not persisted: mutable grounded atoms cannot round-trip
    /// a process boundary, and immutable ones decode as their display form.
    pub fn save_paths(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        self.kernel.backup_paths(path).map(|_| ())
    }

    /// Loads a [`save_paths`](Self::save_paths) file into this space (union
    /// with current contents). Conservatively drops the var-freeness latch:
    /// the file may hold variable-bearing atoms.
    pub fn load_paths(&mut self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        self.kernel.restore_paths(path).map(|_| ())?;
        self.var_free = false;
        self.mutation_gen += 1;
        Ok(())
    }

    /// Bulk-loads atoms across threads, exploiting exactly the property the
    /// wiki describes PathMap by: a prefix-partitioned trie of copy-on-write
    /// nodes, so per-thread private tries merge by structural `join` (shared
    /// subtrees, no deep copies) instead of contending on one writer. Atoms
    /// are `Send + Sync` on the thread-safe base, so the slice shards by
    /// reference. Atoms carrying *mutable* grounded values need the space's
    /// identity registry and take the sequential path; everything else
    /// encodes in parallel. Unencodable atoms count into
    /// [`rejected_atom_count`](Self::rejected_atom_count) exactly as in
    /// [`add`](SpaceMut::add).
    pub fn extend_parallel(&mut self, atoms: &[Atom], threads: usize) {
        let threads = threads.max(1);
        let chunk = atoms.len().div_ceil(threads).max(1);

        struct ShardOut {
            trie: PathMap<()>,
            registry: HashMap<String, Atom>,
            rejected: usize,
            var_free: bool,
            mutable_grounded: Vec<usize>,
        }

        let shards: Vec<ShardOut> = std::thread::scope(|scope| {
            let handles: Vec<_> = atoms
                .chunks(chunk)
                .enumerate()
                .map(|(shard_idx, shard)| {
                    scope.spawn(move || {
                        let mut out = ShardOut {
                            trie: PathMap::new(),
                            registry: HashMap::new(),
                            rejected: 0,
                            var_free: true,
                            mutable_grounded: Vec::new(),
                        };
                        let mut reg = GroundedRegistry::default();
                        for (i, atom) in shard.iter().enumerate() {
                            if contains_mutable_grounded(atom) {
                                out.mutable_grounded.push(shard_idx * chunk + i);
                                continue;
                            }
                            reg.register(atom);
                            let mut vars = Vec::new();
                            let mut bytes = Vec::new();
                            if encode_atom(atom, &mut vars, &mut bytes, &mut GroundedSink::ValueOnly)
                                .is_err()
                            {
                                out.rejected += 1;
                                continue;
                            }
                            if atom.iter().filter_type::<&VariableAtom>().next().is_some() {
                                out.var_free = false;
                            }
                            out.trie.insert(&bytes, ());
                        }
                        out.registry = reg.by_display;
                        out
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let mut sequential = Vec::new();
        // Merge by structural join: measured on 1M atoms this beats the
        // ZipperHead per-region graft pattern decisively (20ms vs 75ms at 16
        // threads), because region discovery must scan the shard tries while
        // join already merges by sharing COW subtrees. The zipper-head pattern
        // stays right where work is prefix-sharded up front (PathMap's own
        // parallel bench writes under artificial shard prefixes).
        for shard in shards {
            self.kernel.btm = self.kernel.btm.join(&shard.trie);
            for (display, atom) in shard.registry {
                self.grounded.by_display.entry(display).or_insert(atom);
            }
            self.rejected_atoms += shard.rejected;
            self.var_free = self.var_free && shard.var_free;
            sequential.extend(shard.mutable_grounded);
        }
        // Bulk load: one generation bump invalidates the indexes and the query
        // table; they rebuild on next use.
        self.mutation_gen += 1;
        for idx in sequential {
            self.add(atoms[idx].clone());
        }
    }

    /// Evaluates a MeTTa expression against this space's `(= lhs rhs)`
    /// equations by running the evaluation AS MM2 exec rewriting inside the
    /// kernel -- the first step of MeTTa-on-MORK evaluation, not just
    /// MeTTa-storage-on-MORK. The semantics implemented is the outermost
    /// term-rewriting fragment of the spec's `metta_call`: a term some
    /// equation matches rewrites to every matching body (all branches kept,
    /// MeTTa's nondeterminism); a term no equation matches returns itself.
    /// This covers accumulator/state-machine style programs (each body's root
    /// is the next redex or a constructor); nested-redex programs need the
    /// congruence lowering (LeaTTa's MorkMM2Lowering models it) -- future
    /// work, shipped nowhere yet.
    ///
    /// Runs inside an O(1) [`fork`](Self::fork), so the live space never sees
    /// the evaluation scaffolding; `max_steps` bounds the exec loop (fuel).
    /// With the `semi-naive` feature the fixpoint matches only each round's
    /// delta.
    pub fn reduce(&self, expr: &Atom, max_steps: usize) -> Vec<Atom> {
        let mut scratch = self.fork();
        // Seed the query term; the rewrite rule is a dormant ((exec 0) ...)
        // re-armed each round by the kernel benches' IC scheduler (an exec is
        // consumed when it fires, so multi-round rewriting needs the meta-rule;
        // this is the exact scaffold the semi-naive example validates). The
        // seed symbol is namespaced to stay clear of program atoms; max_steps
        // is the Peano fuel of the scheduler, one unit per rewrite round.
        let seed = Atom::expr([Atom::sym("mm2-ev"), expr.clone()]);
        scratch.add(seed);
        let mut fuel = String::with_capacity(max_steps * 3 + 1);
        for _ in 0..max_steps {
            fuel.push_str("(S ");
        }
        fuel.push('Z');
        for _ in 0..max_steps {
            fuel.push(')');
        }
        let program = format!(
            r#"
(exec (IC 0 0 {fuel})
               (, (exec (IC $x $y (S $c)) $sp $st)
                  ((exec $x) $p $t))
               (, (exec (IC $y $x $c) $sp $st)
                  (exec (R $x) $p $t)))
((exec 0) (, (mm2-ev $x) (= $x $y)) (, (mm2-ev $y)))
"#
        );
        if scratch.add_sexpr_text(&program).is_err() {
            return Vec::new();
        }
        scratch.step(usize::MAX / 2);

        // Every (mm2-ev <t>) in the dish is a term the rewrite reached; the
        // results are the irreducible ones (no equation matches them), which
        // is exactly metta_call's no-match self-return.
        let reached = scratch.query(&Atom::expr([Atom::sym("mm2-ev"), Atom::var("t")]));
        let t_var = VariableAtom::new("t");
        let mut results = Vec::new();
        for b in reached.iter() {
            let Some(term) = b.resolve(&t_var) else {
                continue;
            };
            let redex_probe = Atom::expr([
                Atom::sym("="),
                term.clone(),
                Atom::var("mm2_reduct"),
            ]);
            if scratch.query(&redex_probe).is_empty() {
                results.push(term);
            }
        }
        results
    }

    /// An O(1) copy-on-write fork: the trie shares structure with `self` until
    /// either side mutates (PathMap node sharing), so branch-and-explore costs
    /// nothing up front -- against the O(N) per-atom copy a space clone
    /// otherwise costs. The fork carries the registry, the var-freeness latch,
    /// and the current column indexes and tabled queries (all copy-on-write).
    pub fn fork(&self) -> MorkSpace {
        MorkSpace {
            kernel: {
                let mut k = MorkKernel::new();
                k.btm = self.kernel.btm.clone();
                k
            },
            common: SpaceCommon::default(),
            grounded: self.grounded.clone(),
            rejected_atoms: self.rejected_atoms,
            column_index_cache: std::sync::RwLock::new(
                self.column_index_cache.read().unwrap().clone(),
            ),
            mutation_gen: self.mutation_gen,
            query_cache: std::sync::RwLock::new(self.query_cache.read().unwrap().clone()),
            var_free: self.var_free,
        }
    }

    /// Trie set algebra (PathMap `join`/`meet`/`subtract`): the operation runs
    /// on the copy-on-write trie structure, sharing every common subtree with
    /// both operands, instead of iterating atoms -- the CeTTa `mork:` algebra
    /// surface, here as destination-mutating Space operations. Declines (returns
    /// false, no change) when either space holds *mutable* grounded atoms:
    /// their per-instance ids are space-local, so the other side's id bytes
    /// would dangle in the merged trie. Immutable grounded atoms are
    /// content-addressed and merge safely.
    pub fn union_with(&mut self, other: &MorkSpace) -> bool {
        self.algebra(other, |a, b| a.join(b))
    }

    /// Keeps exactly the atoms present in both spaces. See [`union_with`](Self::union_with).
    pub fn intersect_with(&mut self, other: &MorkSpace) -> bool {
        self.algebra(other, |a, b| a.meet(b))
    }

    /// Removes every atom present in `other`. See [`union_with`](Self::union_with).
    pub fn subtract_space(&mut self, other: &MorkSpace) -> bool {
        self.algebra(other, |a, b| a.subtract(b))
    }

    fn algebra(
        &mut self,
        other: &MorkSpace,
        op: impl FnOnce(&PathMap<()>, &PathMap<()>) -> PathMap<()>,
    ) -> bool {
        if !self.grounded.by_id.is_empty() || !other.grounded.by_id.is_empty() {
            return false;
        }
        self.kernel.btm = op(&self.kernel.btm, &other.kernel.btm);
        // Adopt the other side's immutable grounded atoms so merged-in marker
        // bytes decode (content-addressed: first instance per display wins).
        for (display, atom) in other.grounded.by_display.iter() {
            self.grounded
                .by_display
                .entry(display.clone())
                .or_insert_with(|| atom.clone());
        }
        self.var_free = self.var_free && other.var_free;
        self.mutation_gen += 1;
        true
    }

    /// A `Send + Sync` read-only snapshot for data-parallel querying: a cheap
    /// copy-on-write clone of the trie that many threads can query concurrently.
    /// MORK's `PathMap` is `Send + Sync`, so this is the parallel querying that
    /// Hyperon's `Rc<RefCell>` spaces (issue #410) cannot express.
    pub fn snapshot(&self) -> MorkSnapshot {
        let gen = self.mutation_gen;
        let indexes = self
            .column_index_cache
            .read()
            .unwrap()
            .iter()
            .filter(|(_, (_, entry_gen))| *entry_gen == gen)
            .map(|(key, (index, _))| (key.clone(), index.clone()))
            .collect();
        MorkSnapshot {
            btm: self.kernel.btm.clone(),
            var_free: self.var_free,
            indexes,
        }
    }
}

/// Adds a binding the way Hyperon's matcher does (matcher.rs `Bindings::from`): a
/// variable bound to another variable is a variable *equality* (so equivalence classes
/// merge and otherwise-equal results collapse instead of multiplying), and anything else
/// is a value binding. Hyperon's insertion operations can split a single binding into
/// several alternatives, so keep the whole set instead of forcing one result.
fn bind_or_equate(set: BindingsSet, var: VariableAtom, atom: Atom) -> BindingsSet {
    match atom {
        Atom::Variable(v) => set.add_var_equality(&var, &v),
        _ => set.add_var_binding(var, atom),
    }
}

/// Decodes the variable at `index` in the current namespace. Namespace 0 is the query,
/// so its variables are the caller's own (`query_vars[index]`), preserving `$n` through
/// the round-trip; other namespaces (matched data atoms / factors) get `v{ns}_{index}`.
fn decode_var(ctx: &DecodeCtx, index: usize) -> Atom {
    if ctx.ns == 0 {
        if let Some(v) = ctx.query_vars.get(index) {
            return Atom::Variable(v.clone());
        }
    }
    Atom::Variable(VariableAtom::new_id(
        format!("v{}_{}", ctx.ns, index),
        ctx.result_id,
    ))
}

/// Live-value post-filter for mutable grounded atoms in the query (e.g. a State
/// cell in a `match` pattern). Each was encoded as a wildcard that matched any
/// atom at its position; keep a result only if the captured atom equals the
/// query atom by *current* value (`Atom`/`State` PartialEq derefs the cell), so
/// a frozen byte image cannot over-match. Shared by the matcher path and the
/// column-index path.
fn apply_live_refs(
    acc: BindingsSet,
    refs: &[(usize, Atom)],
    vars: &[VariableAtom],
) -> BindingsSet {
    if refs.is_empty() {
        return acc;
    }
    let mut filtered = BindingsSet::empty();
    for b in acc {
        let mut keep = true;
        for (idx, ref_atom) in refs {
            let ok = vars
                .get(*idx)
                .and_then(|v| b.resolve(v))
                .is_some_and(|captured| &captured == ref_atom);
            if !ok {
                keep = false;
                break;
            }
        }
        if keep {
            filtered.push(b);
        }
    }
    filtered
}

fn result_var_set<'a>(
    vars: &'a [VariableAtom],
    refs: &[(usize, Atom)],
) -> HashSet<&'a VariableAtom> {
    let hidden: HashSet<usize> = refs.iter().map(|(idx, _)| *idx).collect();
    vars.iter()
        .enumerate()
        .filter_map(|(idx, var)| (!hidden.contains(&idx)).then_some(var))
        .collect()
}

/// Classifies a query for the column-index route: single factor, the fork's
/// admitted fragment, not leading-bound. Returns the classification, the chosen
/// (most selective by longest encoded value) bound position and its value, plus
/// the pattern's variables and mutable-grounded refs.
#[allow(clippy::type_complexity)]
fn classify_index_route(
    query: &Atom,
) -> Option<(
    argindex::Classified,
    usize,
    Vec<u8>,
    Vec<VariableAtom>,
    Vec<(usize, Atom)>,
)> {
    if conjuncts(query).is_some() {
        return None;
    }
    let pattern = wrap_pattern(query)?;
    // Strip the `(, ...)` wrapper: Arity(2) + SymbolSize(1) + b','.
    let classified = argindex::classify_single_factor(&pattern.bytes[3..])?;
    if matches!(
        classified.args.first(),
        Some(argindex::ArgClass::Bound(_))
    ) {
        // Leading argument bound: the primary trie already seeks.
        return None;
    }
    let (pos, value) = classified
        .args
        .iter()
        .enumerate()
        .filter_map(|(i, a)| match a {
            argindex::ArgClass::Bound(v) => Some((i, v.clone())),
            argindex::ArgClass::Free => None,
        })
        .max_by_key(|(_, v)| v.len())?;
    Some((classified, pos, value, pattern.vars, pattern.refs))
}

/// Turns the facts an index seek returned into the matcher-identical
/// `BindingsSet`: residual bound columns filter byte-exact (the store is
/// variable-free under the latch), free columns decode and bind, and the
/// mutable-grounded live-value post-filter applies. Shared by the live space
/// and the snapshot.
fn indexed_facts_to_bindings(
    facts: &[Vec<u8>],
    classified: &argindex::Classified,
    pos: usize,
    vars: &[VariableAtom],
    refs: &[(usize, Atom)],
    mut reg: GroundedRegistry,
    query: &Atom,
) -> BindingsSet {
    let args = &classified.args;
    let ncols = args.len();
    reg.register(query);
    let result_vars = result_var_set(vars, refs);
    let mut set = BindingsSet::empty();
    'fact: for fact in facts {
        let cols = argindex::split_columns(&fact[classified.functor_prefix.len()..], ncols);
        if cols.len() != ncols {
            continue;
        }
        let result_id = next_variable_id();
        let mut acc = BindingsSet::single();
        let mut var_idx = 0usize;
        for (i, arg) in args.iter().enumerate() {
            match arg {
                argindex::ArgClass::Bound(v) => {
                    if i != pos && cols[i] != &v[..] {
                        continue 'fact;
                    }
                }
                argindex::ArgClass::Free => {
                    let mut posn = 0usize;
                    let mut ctx = DecodeCtx {
                        ns: 1,
                        var_counter: 0,
                        grounded: Some(&reg),
                        query_vars: vars,
                        evaluated: None,
                        result_id,
                    };
                    let Some(atom) = decode_atom(cols[i], &mut posn, &mut ctx) else {
                        continue 'fact;
                    };
                    let Some(var) = vars.get(var_idx) else {
                        continue 'fact;
                    };
                    acc = bind_or_equate(acc, var.clone(), atom);
                    var_idx += 1;
                    if acc.is_empty() {
                        continue 'fact;
                    }
                }
            }
        }
        // The var-freeness latch admits this route only on ground stores, so
        // every binding maps a query variable to a ground value and there is
        // nothing to narrow away.
        set.extend(apply_live_refs(acc, refs, vars));
    }
    set
}

/// One binding of one raw match, exactly as the kernel callback saw it: the
/// `(ns, idx)` key, the captured sub-expression's bytes (copied out of the
/// traversal), and the span's owning namespace `env.n` plus NewVar seed `env.v`.
/// Rows replay through [`rows_to_bindings`], which re-decodes with fresh result
/// ids, so a replayed row is indistinguishable from a live match.
#[derive(Clone)]
struct RawBinding {
    key_ns: u8,
    key_idx: u8,
    span: Vec<u8>,
    env_n: u8,
    env_v: u8,
}

/// One raw match: the kernel unifier's bindings in `BTreeMap` iteration order,
/// plus the sort key that makes result order deterministic and Grounding-like:
/// more specific facts (longer exact-byte agreement with the pattern) come
/// first, and equal-specificity rows keep the kernel's trie order. Insertion
/// order is NOT recovered: a set-semantics trie has no insertion history, and a
/// sidecar that tracks one measured a 16x bulk-load regression for an ordering
/// MeTTa leaves unspecified.
#[derive(Clone)]
struct RawRow {
    entries: Vec<RawBinding>,
    fact: Option<Vec<u8>>,
    sequence: usize,
}

/// Runs the kernel matcher and captures every match as a [`RawRow`]. `wrapped`
/// is the `(, ...)`-encoded pattern; the kernel reads it without mutating it
/// (the prepared-query contract).
fn query_btm_rows(btm: &PathMap<()>, wrapped: &[u8]) -> Vec<RawRow> {
    let pat_expr = Expr {
        ptr: wrapped.as_ptr() as *mut u8,
    };
    let pattern_factor = wrapped.get(3..).unwrap_or(wrapped);
    let mut rows = Vec::new();
    MorkKernel::query_multi(btm, pat_expr, |res, loc| {
        let Err(bindings) = res else { return true };
        let loc_span = unsafe { loc.span().as_ref() };
        let sequence = rows.len();
        let entries = bindings
            .iter()
            .map(|(&(key_ns, key_idx), env)| RawBinding {
                key_ns,
                key_idx,
                span: unsafe { env.subsexpr().span().as_ref().unwrap() }.to_vec(),
                env_n: env.n,
                env_v: env.v,
            })
            .collect::<Vec<_>>();
        rows.push(RawRow {
            entries,
            fact: loc_span.map(|span| span.to_vec()),
            sequence,
        });
        true
    });
    // The specificity key exists to order rows, so a 0- or 1-row result skips
    // building it entirely (point lookups stay allocation-lean); a multi-row
    // result derives each key once from the captured fact span.
    if rows.len() > 1 {
        rows.sort_by_cached_key(|row| {
            (
                row.fact
                    .as_deref()
                    .map(|span| specificity_key(pattern_factor, span))
                    .unwrap_or_default(),
                row.sequence,
            )
        });
    }
    rows
}

/// Diagnostic spike: counts kernel callbacks without any materialization.
/// No Vec, no sort, no decode — just counting how many times the kernel fires.
/// Callback always returns true to force full enumeration.
fn query_btm_rows_count_raw(btm: &PathMap<()>, wrapped: &[u8]) -> usize {
    let pat_expr = Expr {
        ptr: wrapped.as_ptr() as *mut u8,
    };
    let mut count = 0usize;
    MorkKernel::query_multi(btm, pat_expr, |_res, _loc| {
        count += 1;
        true
    });
    count
}

/// Shared decode-and-visit core: runs `query_multi` on the kernel, decodes each
/// matched binding, and calls `sink` for every successfully narrowed result.
/// Returns (total_callbacks, sink_calls, errors).
fn decode_and_visit(
    btm: &PathMap<()>,
    wrapped: &[u8],
    vars: &[VariableAtom],
    refs: &[(usize, Atom)],
    reg: &GroundedRegistry,
    query_evaluated: &EvaluatedSpans,
    source_query: Option<&Atom>,
    mut sink: impl FnMut(Bindings) -> bool,
) -> (usize, usize, usize) {
    let pat_expr = Expr {
        ptr: wrapped.as_ptr() as *mut u8,
    };
    let mut total_callbacks = 0usize;
    let mut sink_calls = 0usize;
    let mut errors = 0usize;
    let result_vars = result_var_set(vars, refs);
    MorkKernel::query_multi(btm, pat_expr, |res, loc| {
        total_callbacks += 1;
        let Err(bindings) = res else { return true };
        let loc_span = unsafe { loc.span().as_ref() };
        // Fast path: single-factor fact span narrowing
        let fact_needs_narrowing = loc_span.is_some_and(|f| !span_is_ground(f));
        if let (Some(query), Some(fact), true) = (source_query, loc_span, fact_needs_narrowing) {
            let mut pos = 0usize;
            let mut ctx = DecodeCtx {
                var_counter: 0,
                grounded: Some(reg),
                ns: 1,
                query_vars: vars,
                evaluated: None,
                result_id: next_variable_id(),
            };
            if let Some(fact_atom) = decode_atom(fact, &mut pos, &mut ctx) {
                let mut matched = false;
                for b in match_atoms(&fact_atom, query)
                    .into_iter()
                    .map(|b| b.narrow_vars(&result_vars))
                {
                    matched = true;
                    sink_calls += 1;
                    if !sink(b) {
                        return false;
                    }
                }
                if !matched {
                    errors += 1;
                }
                return true;
            }
        }
        // General path: decode entries from bindings map
        let mut acc = BindingsSet::single();
        let result_id = next_variable_id();
        for (&(key_ns, key_idx), env) in bindings.iter() {
            let mut pos = 0usize;
            let mut ctx = DecodeCtx {
                var_counter: env.v as usize,
                grounded: Some(reg),
                ns: env.n,
                query_vars: vars,
                evaluated: (env.n == 0).then_some(query_evaluated),
                result_id,
            };
            let Some(atom) = decode_atom(
                unsafe { env.subsexpr().span().as_ref().unwrap() },
                &mut pos,
                &mut ctx,
            ) else {
                continue;
            };
            let var = if key_ns == 0 {
                match vars.get(key_idx as usize) {
                    Some(v) => v.clone(),
                    None => continue,
                }
            } else {
                VariableAtom::new_id(format!("v{}_{}", key_ns, key_idx), result_id)
            };
            acc = bind_or_equate(acc, var, atom);
            if acc.is_empty() {
                break;
            }
        }
        let needs_narrowing = bindings
            .iter()
            .any(|(k, env)| k.0 != 0 || !span_is_ground(unsafe { env.subsexpr().span().as_ref().unwrap() }));
        let narrowed = apply_live_refs(acc, refs, vars)
            .into_iter()
            .map(|b| b.narrow_vars(&result_vars));
        let mut matched = false;
        for b in narrowed {
            matched = true;
            sink_calls += 1;
            if !sink(b) {
                return false;
            }
        }
        if !matched {
            errors += 1;
        }
        true
    });
    (total_callbacks, sink_calls, errors)
}

/// Diagnostic spike: counts kernel callbacks and decodes each binding inline,
/// then drops it immediately. No Vec accumulation, no sort, no BindingsSet return.
/// Returns (total_callbacks, decode_ok, decode_errors).
fn query_btm_rows_count_decoded(
    btm: &PathMap<()>,
    wrapped: &[u8],
    vars: &[VariableAtom],
    refs: &[(usize, Atom)],
    reg: &GroundedRegistry,
    query_evaluated: &EvaluatedSpans,
    source_query: Option<&Atom>,
) -> (usize, usize, usize) {
    decode_and_visit(btm, wrapped, vars, refs, reg, query_evaluated, source_query, |_| true)
}

/// Decodes raw matcher rows into the caller's `BindingsSet`.
///
/// Single-factor rows carry the matched fact span. Re-decode that fact, run it
/// through Hyperon's matcher, then narrow to the query variables. That follows
/// `GroundingSpace`'s `AtomIndex::query` path and prevents matched-fact helper
/// variables from leaking into recursive interpreter bindings. Raw row replay
/// remains the fallback for cases without a fact span.
///
/// Every raw replay row gets one fresh id: every data variable decoded below
/// shares it, so the result is internally coreferent by name yet globally
/// distinct from every other result and query call -- replayed rows included.
/// Without this, a data var `v1_0` from one match aliased a `v1_0` from a
/// deeper recursion's match, and the interpreter's threaded bindings collapsed
/// the branch to empty (b2 backchaining: mortal<-human<-And).
fn rows_to_bindings(
    rows: &[RawRow],
    vars: &[VariableAtom],
    refs: &[(usize, Atom)],
    reg: &GroundedRegistry,
    query_evaluated: &EvaluatedSpans,
    source_query: Option<&Atom>,
) -> BindingsSet {
    let mut set = BindingsSet::empty();
    let result_vars = result_var_set(vars, refs);
    for row in rows {
        let fact_needs_narrowing = row.fact.as_deref().is_some_and(|f| !span_is_ground(f));
        if let (Some(query), Some(fact), true) = (source_query, row.fact.as_deref(), fact_needs_narrowing) {
            let mut pos = 0usize;
            let mut ctx = DecodeCtx {
                var_counter: 0,
                grounded: Some(reg),
                ns: 1,
                query_vars: vars,
                evaluated: None,
                result_id: next_variable_id(),
            };
            if let Some(fact_atom) = decode_atom(fact, &mut pos, &mut ctx) {
                set.extend(match_atoms(&fact_atom, query).map(|bindings| {
                    bindings.narrow_vars(&result_vars)
                }));
                continue;
            }
        }
        let mut acc = BindingsSet::single();
        let result_id = next_variable_id();
        // The key (ns, idx) names a variable by its namespace -- 0 is the query
        // pattern, >=1 a matched data atom / factor -- and its index; the bound
        // value's own variables live in the value's namespace `env.n`. Decode
        // names a variable `v{ns}_{idx}`, so a data variable and a pattern
        // variable can never collide. (Without this, evaluating (Add (S $n) Z)
        // bound the data variable $x to the compound (S $n); both decoded to
        // `v0`, yielding the spurious occurs-failing binding `v0 <- (S v0)` that
        // dropped the whole result and stalled b1_equal_chain.) Query variables
        // (ns 0) bind their real names so the result is in terms the caller
        // asked for; data/factor variables bind `v{ns}_{idx}` for Hyperon's
        // transitive `resolve` to chase.
        for b in &row.entries {
            let mut pos = 0usize;
            // Seed the NewVar counter with `env.v`: the count of NewVars
            // preceding this span in its base atom (maintained by
            // ExprEnv::args). A captured sub-span is carved from the middle of
            // an atom, so its `VarRef(i)` bytes index variables in the atom's
            // *global* scope, and a `NewVar` inside it is the (env.v)-th
            // variable of that scope, not the 0th. Seeding from env.v aligns
            // NewVar names with VarRef names, so a variable's binder and body
            // occurrences share one name (consistent VBTO).
            let mut ctx = DecodeCtx {
                var_counter: b.env_v as usize,
                grounded: Some(reg),
                ns: b.env_n,
                query_vars: vars,
                evaluated: (b.env_n == 0).then_some(query_evaluated),
                result_id,
            };
            let Some(atom) = decode_atom(&b.span, &mut pos, &mut ctx) else {
                continue;
            };
            let var = if b.key_ns == 0 {
                match vars.get(b.key_idx as usize) {
                    Some(v) => v.clone(),
                    None => continue,
                }
            } else {
                VariableAtom::new_id(format!("v{}_{}", b.key_ns, b.key_idx), result_id)
            };
            acc = bind_or_equate(acc, var, atom);
            if acc.is_empty() {
                break;
            }
        }
        // e3: `&state-active` matches the goal whose cell currently holds that
        // value, not one mutated since it was stored.
        //
        // A row whose keys are all query-namespace and whose captured spans are
        // all ground binds nothing but query variables to ground values, so the
        // narrow is a no-op and skipped (it costs a per-result equivalence-class
        // walk that dominated warm ground lookups).
        let needs_narrowing = row
            .entries
            .iter()
            .any(|b| b.key_ns != 0 || !span_is_ground(&b.span));
        if needs_narrowing {
            set.extend(
                apply_live_refs(acc, refs, vars)
                    .into_iter()
                    .map(|bindings| bindings.narrow_vars(&result_vars)),
            );
        } else {
            set.extend(apply_live_refs(acc, refs, vars));
        }
    }
    set
}

/// The byte-level query against a bare trie, shared by `MorkSpace` and the
/// `Send + Sync` `MorkSnapshot`.
fn query_btm(btm: &PathMap<()>, query: &Atom, grounded: Option<&GroundedRegistry>) -> BindingsSet {
    // The empty conjunction `(,)` is a fold over zero sub-queries: one empty binding.
    if conjuncts(query).is_some_and(|qs| qs.is_empty()) {
        return BindingsSet::single();
    }
    let Some(pattern) = wrap_pattern(query) else {
        return BindingsSet::empty();
    };
    // Decode must recover grounded atoms that appear in the query itself (the `4` in
    // `(= (sqr 4) $X)`), not only previously-stored ones, so a data-side binding like
    // $x<-4 round-trips as the grounded Number rather than a bare symbol "4" that `*`
    // cannot reduce. Merge the query's grounded atoms with the space registry here.
    let mut reg = grounded.cloned().unwrap_or_default();
    reg.register(query);
    let rows = query_btm_rows(btm, &pattern.bytes);
    rows_to_bindings(
        &rows,
        &pattern.vars,
        &pattern.refs,
        &reg,
        &pattern.evaluated,
        conjuncts(query).is_none().then_some(query),
    )
}

/// A `Send + Sync` read-only snapshot of a space's atoms (a copy-on-write clone of
/// the MORK trie). Construct with [`MorkSpace::snapshot`]; share one across threads
/// (e.g. `Arc<MorkSnapshot>`) for concurrent queries.
pub struct MorkSnapshot {
    btm: PathMap<()>,
    /// Carried var-freeness latch (see [`MorkSpace`]): gates the byte-level
    /// fast paths on this snapshot.
    var_free: bool,
    /// The space's fresh column indexes at snapshot time (copy-on-write clones,
    /// so carrying them is O(cached indexes), not O(data)). A selective query
    /// on the snapshot seeks these exactly like the live space; a shape with no
    /// carried index falls through to the matcher (snapshots never build).
    indexes: HashMap<(Vec<u8>, usize, usize), PathMap<()>>,
}

/// A query pattern encoded once for repeated execution, so the per-query encode (the
/// `wrap_pattern` allocation and atom walk) is paid a single time instead of every call.
/// This is the in-process analog of the PeTTa<->MORK FFI bridge's prepared-count handle.
/// Build with [`MorkSnapshot::prepare`]; the kernel reads the bytes without mutating them, so
/// run it as many times as needed (hold one per thread for parallel use).
pub struct PreparedQuery {
    wrapped: Vec<u8>,
    // Kept for a future binding-returning `query_prepared`; `count_prepared` needs only `wrapped`.
    #[allow(dead_code)]
    vars: Vec<VariableAtom>,
    #[allow(dead_code)]
    refs: Vec<(usize, Atom)>,
    #[allow(dead_code)]
    evaluated: EvaluatedSpans,
}

impl MorkSnapshot {
    /// Query the snapshot; safe to call concurrently from many threads. A
    /// selective single-factor query seeks the carried column indexes (frozen
    /// at snapshot time); everything else runs the matcher.
    pub fn query(&self, query: &Atom) -> BindingsSet {
        if self.var_free {
            if let Some((classified, pos, value, vars, refs)) = classify_index_route(query) {
                let ncols = classified.args.len();
                if let Some(index) =
                    self.indexes
                        .get(&(classified.functor_prefix.clone(), ncols, pos))
                {
                    let facts = argindex::arg_index_seek(
                        index,
                        &classified.functor_prefix,
                        ncols,
                        pos,
                        &value,
                    );
                    return indexed_facts_to_bindings(
                        &facts,
                        &classified,
                        pos,
                        &vars,
                        &refs,
                        GroundedRegistry::default(),
                        query,
                    );
                }
            }
        }
        query_btm(&self.btm, query, None)
    }

    /// Counts atoms matching `query` through the trie read path only, without decoding
    /// bindings. Isolates the read-traversal cost from the per-result decode + `Bindings`
    /// allocation, so the two can be compared under parallel load.
    ///
    /// With the `factorized-aggregate` feature, a conjunctive count folds the join
    /// in O(N^fhtw) instead of enumerating it (see [`factorized_count`]).
    pub fn count_matches(&self, query: &Atom) -> usize {
        let Some(mut pattern) = wrap_pattern(query) else {
            return 0;
        };
        if self.var_free {
            if let Some(count) = factorized_count(&self.btm, &pattern.bytes) {
                return count;
            }
        }
        let pat_expr = Expr {
            ptr: pattern.bytes.as_mut_ptr(),
        };
        MorkKernel::query_multi(&self.btm, pat_expr, |_res, _loc| true)
    }

    /// Encodes `query` into reusable bytes once, so the same pattern can be run many times
    /// without re-paying the per-query encode (the `wrap_pattern` allocation + atom walk).
    /// This is the prepared-query model proven by the PeTTa<->MORK FFI bridge
    /// (`prepare_count` / `count_prepared`): hoist the encode out of the hot loop. Returns
    /// `None` for an unencodable pattern (symbol/arity over 63).
    pub fn prepare(&self, query: &Atom) -> Option<PreparedQuery> {
        let pattern = wrap_pattern(query)?;
        Some(PreparedQuery {
            wrapped: pattern.bytes,
            vars: pattern.vars,
            refs: pattern.refs,
            evaluated: pattern.evaluated,
        })
    }

    /// Counts matches of a [`PreparedQuery`] reusing its encoded bytes. The kernel reads the
    /// pattern without mutating it (the FFI bridge runs a prepared count many times over one
    /// handle), so the bytes are reused across calls; hold one `PreparedQuery` per thread.
    pub fn count_prepared(&self, prepared: &PreparedQuery) -> usize {
        if self.var_free {
            if let Some(count) = factorized_count(&self.btm, &prepared.wrapped) {
                return count;
            }
        }
        let pat_expr = Expr {
            ptr: prepared.wrapped.as_ptr() as *mut u8,
        };
        MorkKernel::query_multi(&self.btm, pat_expr, |_res, _loc| true)
    }

    /// Number of atoms in the snapshot.
    pub fn len(&self) -> usize {
        self.btm.val_count()
    }
}

/// Counts a conjunctive query's join by factorized aggregation instead of
/// enumeration: parse the encoded `(, f1 .. fn)` body into factors, decompose
/// their hypergraph, and fold counts up the join tree -- O(N^fhtw) where
/// enumeration is O(join output) (fac17's asymptotic win; a two-factor product
/// join counts in linear time while its output is quadratic). Counting matches
/// keeps every variable, so fac18's full-projection condition holds by
/// construction, and there is no grouping template (fac20). Requires the
/// `factorized-aggregate` feature and at least two factors (a single factor
/// already counts in O(matches)); on any decline (`None`) the caller runs the
/// enumerating count, whose answer is byte-identical (the kernel's
/// factorized-aggregate fuzz differential, re-checked here against the
/// enumerating count in this crate's proptest suite).
fn factorized_count(btm: &PathMap<()>, wrapped: &[u8]) -> Option<usize> {
    if !cfg!(feature = "factorized-aggregate") {
        return None;
    }
    let (factors, nvars) = mork::zipper_join::parse_body_factors(wrapped)?;
    if factors.len() < 2 {
        return None;
    }
    // The aggregate fold materializes variable values per top-level column; a
    // variable nested inside a compound column never materializes one (the
    // kernel's bag-key encoder panics on exactly that shape), so admit only
    // factors whose every column is a top-level variable or fully ground.
    if factors
        .iter()
        .any(|f| f.cols.iter().any(|c| c.is_nonground_compound()))
    {
        return None;
    }
    mork::ghd::ghd_aggregate_auto::<u64>(btm, &factors, nvars, |_| 1).map(|c| c as usize)
}

/// Returns the conjuncts of a top-level `(, q1 .. qn)` query (hyperon-space's
/// `COMMA_SYMBOL` contract: "Query may include sub-queries glued by [the comma]"),
/// or `None` for an ordinary single-pattern query.
fn conjuncts(query: &Atom) -> Option<&[Atom]> {
    let Atom::Expression(e) = query else {
        return None;
    };
    match e.children().first() {
        Some(Atom::Symbol(s)) if s.name() == "," => Some(&e.children()[1..]),
        _ => None,
    }
}

/// Encodes `query` into the `(, <p1> .. <pn>)` multi-pattern form `query_multi`
/// expects, returning the bytes, the variables in introduction order, and the
/// mutable-grounded positions `(var index, atom)` to post-filter by live value
/// (empty for the common case).
///
/// A top-level `(, q1 .. qn)` query encodes each conjunct as its own factor of
/// the kernel's multi-pattern query, sharing one variable introduction order, so
/// a variable used in two conjuncts becomes a `VarRef` across factors -- a join
/// variable the kernel's worst-case-optimal join matches natively, instead of the
/// per-conjunct query-and-thread loop (and its intermediate-product blowup) that
/// an interpreter-side conjunction costs.
fn wrap_pattern(query: &Atom) -> Option<WrappedPattern> {
    let mut vars = Vec::new();
    let mut refs = Vec::new();
    let mut evaluated = EvaluatedSpans::default();
    let mut wrapped = Vec::with_capacity(64);
    let mut sink = GroundedSink::Query(&mut refs);
    match conjuncts(query) {
        Some(qs) if !qs.is_empty() => {
            // The wrapper's arity field also holds the `,` head, so at most
            // MAX_FIELD - 1 conjuncts encode.
            if qs.len() > MAX_FIELD - 1 {
                return None;
            }
            wrapped.push(item_byte(Tag::Arity((1 + qs.len()) as u8)));
            wrapped.push(item_byte(Tag::SymbolSize(1)));
            wrapped.push(b',');
            for q in qs {
                if encode_atom_collecting_evaluated(
                    q,
                    &mut vars,
                    &mut wrapped,
                    &mut sink,
                    &mut evaluated,
                )
                .is_err()
                {
                    return None;
                }
            }
        }
        _ => {
            wrapped.push(item_byte(Tag::Arity(2)));
            wrapped.push(item_byte(Tag::SymbolSize(1)));
            wrapped.push(b',');
            if encode_atom_collecting_evaluated(
                query,
                &mut vars,
                &mut wrapped,
                &mut sink,
                &mut evaluated,
            )
            .is_err()
            {
                return None;
            }
        }
    }
    Some(WrappedPattern {
        bytes: wrapped,
        vars,
        refs,
        evaluated,
    })
}

/// A hash-prefix-sharded MORK space for data-parallel whole-space sweeps -- the
/// ShardZipper symbolic-CPU path (Goertzel 2025). Atoms are partitioned across
/// `n_shards` PathMap tries by a hash of their byte encoding; each shard is a
/// locally-sweepable sub-trie, and a whole-space match-count sweeps all shards in
/// parallel with rayon. (A ground point query lands in one shard; a pattern that can
/// match anywhere sweeps all shards, which is where the parallelism pays.)
pub struct ShardedMorkSpace {
    shards: Vec<PathMap<()>>,
    rejected_atoms: usize,
}

impl ShardedMorkSpace {
    /// Creates an empty sharded space with `n_shards` shards (at least one).
    pub fn new(n_shards: usize) -> Self {
        Self {
            shards: (0..n_shards.max(1)).map(|_| PathMap::new()).collect(),
            rejected_atoms: 0,
        }
    }

    fn shard_of(&self, bytes: &[u8]) -> usize {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        bytes.hash(&mut h);
        (h.finish() as usize) % self.shards.len()
    }

    /// Adds an atom to its hash-determined shard. Returns false on an unencodable atom.
    /// A sharded shard has no registry, so a mutable grounded atom falls back to its
    /// (unstable) display key; sharded sweeps are for immutable, content-addressed data.
    pub fn add(&mut self, atom: &Atom) -> bool {
        let mut vars = Vec::new();
        let mut bytes = Vec::new();
        if encode_atom(atom, &mut vars, &mut bytes, &mut GroundedSink::ValueOnly).is_err() {
            self.rejected_atoms += 1;
            return false;
        }
        let s = self.shard_of(&bytes);
        self.shards[s].insert(&bytes, ());
        true
    }

    /// Total atoms across all shards.
    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.val_count()).sum()
    }

    /// Whether the space holds no atoms.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of atoms rejected by `add` because they do not fit MORK's byte encoding.
    pub fn rejected_atom_count(&self) -> usize {
        self.rejected_atoms
    }

    /// Number of shards.
    pub fn shards(&self) -> usize {
        self.shards.len()
    }

    /// Counts atoms matching `pattern`, sweeping every shard in parallel (rayon).
    pub fn par_count_matches(&self, pattern: &Atom) -> usize {
        use rayon::prelude::*;
        let Some(mut pattern) = wrap_pattern(pattern) else {
            return 0;
        };
        // Pass the pattern-buffer address as a usize (Send) and rebuild the Expr in
        // each task; the buffer is alive for the whole sweep and read-only.
        let pat_ptr = pattern.bytes.as_mut_ptr() as usize;
        self.shards
            .par_iter()
            .map(|shard| {
                let pat_expr = Expr {
                    ptr: pat_ptr as *mut u8,
                };
                MorkKernel::query_multi(shard, pat_expr, |_, _| true)
            })
            .sum()
    }

    /// Sequential baseline of [`par_count_matches`].
    pub fn count_matches(&self, pattern: &Atom) -> usize {
        let Some(mut pattern) = wrap_pattern(pattern) else {
            return 0;
        };
        let pat_expr = Expr {
            ptr: pattern.bytes.as_mut_ptr(),
        };
        self.shards
            .iter()
            .map(|shard| MorkKernel::query_multi(shard, pat_expr, |_, _| true))
            .sum()
    }
}

impl Default for MorkSpace {
    fn default() -> Self {
        Self::new()
    }
}

/// How `encode_atom` handles a *mutable* grounded atom (`Grounded::is_mutable`), which is
/// not content-addressable so cannot just be written as its display bytes.
enum GroundedSink<'a> {
    /// Storing into the trie: intern the mutable atom by identity and embed its id, and
    /// register immutable grounded atoms by display for decode.
    Store(&'a mut GroundedRegistry),
    /// Encoding a query pattern: replace the mutable atom with a fresh wildcard variable and
    /// record `(var index, the atom)` so [`query_btm`] can post-filter matches by live value.
    Query(&'a mut Vec<(usize, Atom)>),
    /// No mutable-atom support (sharded shards, removal): fall back to the display key, which is
    /// unstable for a mutable atom but the only option without a registry to intern into.
    ValueOnly,
}

/// Walks a Hyperon `Atom` into MORK's preorder byte encoding, recording variables
/// in first-occurrence order (`NewVar` introduces, later occurrences `VarRef` back).
/// Errors when a symbol or arity exceeds MORK's 63 limit. `sink` decides how a mutable
/// grounded atom (a `State` cell) is handled: interned by identity when storing, turned
/// into a post-filtered wildcard when querying.
fn encode_atom(
    atom: &Atom,
    vars: &mut Vec<VariableAtom>,
    out: &mut Vec<u8>,
    sink: &mut GroundedSink,
) -> Result<(), ()> {
    encode_atom_collecting_evaluated(atom, vars, out, sink, &mut EvaluatedSpans::default())
}

fn encode_atom_collecting_evaluated(
    atom: &Atom,
    vars: &mut Vec<VariableAtom>,
    out: &mut Vec<u8>,
    sink: &mut GroundedSink,
    evaluated: &mut EvaluatedSpans,
) -> Result<(), ()> {
    match atom {
        Atom::Symbol(s) => encode_symbol(s.name(), out),
        Atom::Grounded(g) if g.as_grounded().is_mutable() => match sink {
            // Stored: address the cell by a fresh identity id; the live `Atom` goes in the registry.
            GroundedSink::Store(reg) => encode_grounded_ref(reg.intern_mutable(atom), out),
            // Queried: a wildcard that matches any atom at this position, recorded for the live-value
            // post-filter. Its NewVar slot is a real (fresh, unique) query var so var indexing stays
            // aligned; the caller never reads it, the post-filter does.
            GroundedSink::Query(refs) => {
                out.push(item_byte(Tag::NewVar));
                vars.push(VariableAtom::new("state").make_unique());
                refs.push((vars.len() - 1, atom.clone()));
                Ok(())
            }
            GroundedSink::ValueOnly => encode_grounded_value(&g.to_string(), out),
        },
        Atom::Grounded(g) => {
            // Immutable grounded atom: content-addressed by display. Register it on the store path
            // so decode can rebuild the exact instance (a query path registers via query_btm).
            if let GroundedSink::Store(reg) = sink {
                reg.register(atom);
            }
            encode_grounded_value(&g.to_string(), out)
        }
        Atom::Expression(e) => {
            let children = e.children();
            if children.len() > MAX_FIELD {
                return Err(());
            }
            let start = out.len();
            out.push(item_byte(Tag::Arity(children.len() as u8)));
            for child in children {
                encode_atom_collecting_evaluated(child, vars, out, sink, evaluated)?;
            }
            if e.is_evaluated() {
                evaluated.insert(&out[start..]);
            }
            Ok(())
        }
        Atom::Variable(v) => {
            match vars.iter().position(|x| x == v) {
                Some(i) if i <= MAX_FIELD => out.push(item_byte(Tag::VarRef(i as u8))),
                Some(_) => return Err(()),
                None => {
                    out.push(item_byte(Tag::NewVar));
                    vars.push(v.clone());
                }
            }
            Ok(())
        }
    }
}

fn encode_symbol(name: &str, out: &mut Vec<u8>) -> Result<(), ()> {
    let bytes = name.as_bytes();
    if bytes.is_empty() || bytes.len() > MAX_FIELD {
        return Err(());
    }
    out.push(item_byte(Tag::SymbolSize(bytes.len() as u8)));
    out.extend_from_slice(bytes);
    Ok(())
}

/// Encodes an immutable grounded atom as a marker symbol: [`GROUNDED_MARK`] followed by the
/// atom's display string. `decode_atom` recognises the marker and rebuilds the original `Atom`
/// from the [`GroundedRegistry`]. The display plus marker must fit MORK's 63-byte symbol field.
fn encode_grounded_value(display: &str, out: &mut Vec<u8>) -> Result<(), ()> {
    let bytes = display.as_bytes();
    if bytes.is_empty() || bytes.len() + 1 > MAX_FIELD {
        return Err(());
    }
    out.push(item_byte(Tag::SymbolSize((bytes.len() + 1) as u8)));
    out.push(GROUNDED_MARK);
    out.extend_from_slice(bytes);
    Ok(())
}

/// Encodes a mutable grounded atom by identity: [`GROUNDED_REF_MARK`] followed by the 8-byte
/// registry id. `decode_atom` rebuilds the live cell via [`GroundedRegistry::get_by_id`].
fn encode_grounded_ref(id: u64, out: &mut Vec<u8>) -> Result<(), ()> {
    let id_bytes = id.to_le_bytes();
    out.push(item_byte(Tag::SymbolSize((id_bytes.len() + 1) as u8)));
    out.push(GROUNDED_REF_MARK);
    out.extend_from_slice(&id_bytes);
    Ok(())
}

/// Walks MORK's preorder byte encoding back into a Hyperon `Atom`. `pos` advances
/// past the consumed bytes. Returns `None` on a malformed/short buffer.
fn decode_atom(bytes: &[u8], pos: &mut usize, ctx: &mut DecodeCtx) -> Option<Atom> {
    let start = *pos;
    let tag = byte_item(*bytes.get(*pos)?);
    *pos += 1;
    match tag {
        Tag::SymbolSize(s) => {
            let s = s as usize;
            let end = pos.checked_add(s)?;
            let raw = bytes.get(*pos..end)?;
            *pos = end;
            if let Some((&GROUNDED_MARK, disp_bytes)) = raw.split_first() {
                let disp = std::str::from_utf8(disp_bytes).ok()?;
                // Rebuild the immutable grounded atom from the registry; without one (snapshot or
                // sharded shard) fall back to a bare symbol of the display string.
                return Some(
                    ctx.grounded
                        .and_then(|reg| reg.get(disp))
                        .unwrap_or_else(|| Atom::sym(disp)),
                );
            }
            if let Some((&GROUNDED_REF_MARK, id_bytes)) = raw.split_first() {
                // A mutable grounded atom (a State cell): the 8 id bytes index the registry's
                // identity table, returning an Rc-sharing clone whose live value reflects any
                // change-state! since it was stored. Without a registry, a readable placeholder.
                let id = u64::from_le_bytes(id_bytes.try_into().ok()?);
                return Some(
                    ctx.grounded
                        .and_then(|reg| reg.get_by_id(id))
                        .unwrap_or_else(|| Atom::sym(format!("<state-{id}>"))),
                );
            }
            let name = std::str::from_utf8(raw).ok()?;
            Some(Atom::sym(name))
        }
        Tag::Arity(k) => {
            let mut children = Vec::with_capacity(k as usize);
            for _ in 0..k {
                children.push(decode_atom(bytes, pos, ctx)?);
            }
            let mut atom = Atom::expr(children);
            if ctx
                .evaluated
                .is_some_and(|evaluated| evaluated.contains(&bytes[start..*pos]))
            {
                if let Atom::Expression(expr) = &mut atom {
                    expr.set_evaluated();
                }
            }
            Some(atom)
        }
        // The n-th introduced variable and its back-references share one name, so the
        // coreference in stored atoms like (-> (→ $p $q) $p $q) survives the round-trip.
        Tag::NewVar => {
            let idx = ctx.var_counter;
            ctx.var_counter += 1;
            Some(decode_var(ctx, idx))
        }
        Tag::VarRef(i) => Some(decode_var(ctx, i as usize)),
    }
}

impl Space for MorkSpace {
    fn common(&self) -> FlexRef<'_, SpaceCommon> {
        FlexRef::from_simple(&self.common)
    }

    fn query(&self, query: &Atom) -> BindingsSet {
        self.query_inner(query)
    }

    fn visit_query(
        &self,
        query: &Atom,
        callback: &mut dyn FnMut(Bindings) -> bool,
    ) -> usize {
        if conjuncts(query).is_some_and(|qs| qs.is_empty()) {
            return 0;
        }
        let Some(pattern) = wrap_pattern(query) else {
            return 0;
        };
        let mut reg = self.grounded.clone();
        reg.register(query);
        let mut count = 0usize;
        decode_and_visit(
            &self.kernel.btm,
            &pattern.bytes,
            &pattern.vars,
            &pattern.refs,
            &reg,
            &pattern.evaluated,
            conjuncts(query).is_none().then_some(query),
            |b| {
                count += 1;
                callback(b)
            },
        );
        count
    }

    fn atom_count(&self) -> Option<usize> {
        Some(self.len())
    }

    fn visit(&self, v: &mut dyn SpaceVisitor) -> Result<(), ()> {
        use pathmap::zipper::{ZipperIteration, ZipperMoving};
        let mut rz = self.kernel.btm.read_zipper();
        while rz.to_next_val() {
            let atom_bytes = rz.path();
            let mut pos = 0;
            let mut ctx = DecodeCtx {
                ns: 1,
                var_counter: 0,
                grounded: Some(&self.grounded),
                query_vars: &[],
                evaluated: None,
                result_id: next_variable_id(),
            };
            if let Some(atom) = decode_atom(atom_bytes, &mut pos, &mut ctx) {
                v.accept(Cow::Owned(atom));
            }
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl SpaceMut for MorkSpace {
    fn add(&mut self, atom: Atom) {
        let mut vars = Vec::new();
        let mut bytes = Vec::new();
        // Store sink: registers immutable grounded atoms by display and interns mutable cells
        // (State) by identity into `self.grounded` as it walks the atom.
        let mut sink = GroundedSink::Store(&mut self.grounded);
        if encode_atom(&atom, &mut vars, &mut bytes, &mut sink).is_ok() {
            if atom.iter().filter_type::<&VariableAtom>().next().is_some() {
                self.var_free = false;
            }
            self.update_column_indexes(&bytes, true);
            self.kernel.btm.insert(&bytes, ());
        } else {
            self.rejected_atoms += 1;
        }
    }

    fn remove(&mut self, atom: &Atom) -> bool {
        let mut vars = Vec::new();
        let mut bytes = Vec::new();
        // ValueOnly: removal matches stored bytes by content. An atom containing a mutable cell
        // was stored by identity id, so it cannot be removed by value here (not exercised; the
        // interpreter retracts via the cell, not by re-encoding the State).
        if encode_atom(atom, &mut vars, &mut bytes, &mut GroundedSink::ValueOnly).is_err() {
            return false;
        }
        self.update_column_indexes(&bytes, false);
        self.kernel.btm.remove(&bytes).is_some()
    }

    fn replace(&mut self, from: &Atom, to: Atom) -> bool {
        if self.remove(from) {
            self.add(to);
            true
        } else {
            false
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl std::fmt::Debug for MorkSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MorkSpace({} atoms)", self.len())
    }
}

impl std::fmt::Display for MorkSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MorkSpace({} atoms)", self.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyperon::space::grounding::GroundingSpace;
    use hyperon_atom::Atom;
    use proptest::prelude::*;
    use static_assertions::assert_impl_all;
    use std::cell::RefCell;
    use std::collections::{BTreeSet, HashMap, HashSet};
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::rc::Rc;

    /// The WILLIAM report surfaces the shared rule spine as one readable maximal
    /// pattern with the exact gain formula, and never returns nested prefixes of the
    /// same chain.
    #[test]
    fn frequent_subpatterns_reports_maximal_readable_patterns() {
        let mut space = MorkSpace::new();
        for i in 0..40u32 {
            space.add(Atom::expr([
                Atom::sym("rule"),
                Atom::expr([Atom::sym("when"), Atom::sym("gate")]),
                Atom::sym(format!("a{i:02}")),
            ]));
        }
        space.add(Atom::expr([Atom::sym("fact"), Atom::sym("solo")]));

        let ref_cost = crate::william::REF_COST;
        let report = space.frequent_subpatterns(4, ref_cost);
        assert!(!report.is_empty());
        let (rendered, gain) = &report[0];
        assert_eq!(rendered, "(rule (when gate) …)");
        // gain = (count-1)*len - count*ref_cost with count=40 and the spine's encoded
        // length: arity byte + "rule" symbol (5) + arity byte + "when" symbol (5) +
        // "gate" symbol (5) = 17 bytes.
        assert_eq!(*gain, 39 * 17 - 40 * ref_cost);
        // Maximal: no reported pattern renders as a prefix chain of another (each is a
        // distinct structure).
        for (i, (a, _)) in report.iter().enumerate() {
            for (j, (b, _)) in report.iter().enumerate() {
                if i != j {
                    assert!(!a.trim_end_matches([')', '…', ' ']).is_empty());
                    assert_ne!(a, b);
                }
            }
        }
    }

    /// A prepared query encodes once and is reused across calls; it must give the same count
    /// as `count_matches`/`query` on every call (the kernel reads the pattern without mutating
    /// it), for both a point pattern and a wildcard.
    #[test]
    fn prepared_count_is_correct_and_reuse_stable() {
        let mut space = MorkSpace::new();
        for i in 0..200u32 {
            space.add(Atom::expr([Atom::sym("e"), Atom::sym(format!("n{i}")), Atom::sym(format!("n{}", i + 1))]));
        }
        let snap = space.snapshot();

        let point = Atom::expr([Atom::sym("e"), Atom::sym("n100"), Atom::var("d")]);
        let pp = snap.prepare(&point).unwrap();
        for _ in 0..8 {
            assert_eq!(snap.count_prepared(&pp), 1, "prepared point count must be stable across reuse");
            assert_eq!(snap.count_matches(&point), 1);
            assert_eq!(snap.query(&point).len(), 1);
        }

        let wild = Atom::expr([Atom::sym("e"), Atom::var("a"), Atom::var("b")]);
        let wp = snap.prepare(&wild).unwrap();
        for _ in 0..8 {
            assert_eq!(snap.count_prepared(&wp), 200, "prepared wildcard count must be stable across reuse");
        }
    }

    fn parent(a: &str, b: &str) -> Atom {
        Atom::expr([Atom::sym("parent"), Atom::sym(a), Atom::sym(b)])
    }

    /// A minimal mutable grounded cell mirroring Hyperon's `StateAtom`: an `Rc<RefCell>`
    /// whose value can change in place, with by-current-value equality and `is_mutable`
    /// true. Lets the codec's mutable-grounded path be tested without the hyperon stdlib.
    #[derive(Clone, Debug)]
    struct MutCell(std::sync::Arc<std::sync::RwLock<i64>>);
    impl MutCell {
        fn new(v: i64) -> Self {
            Self(std::sync::Arc::new(std::sync::RwLock::new(v)))
        }
        fn set(&self, v: i64) {
            *self.0.write().unwrap() = v;
        }
    }
    impl std::fmt::Display for MutCell {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "(Cell {})", self.0.read().unwrap())
        }
    }
    impl PartialEq for MutCell {
        fn eq(&self, other: &Self) -> bool {
            *self.0.read().unwrap() == *other.0.read().unwrap()
        }
    }
    impl Grounded for MutCell {
        fn type_(&self) -> Atom {
            Atom::sym("Cell")
        }
        fn is_mutable(&self) -> bool {
            true
        }
    }

    fn resolve(b: &Bindings, name: &str) -> Option<Atom> {
        b.resolve(&VariableAtom::new(name))
    }

    fn evaluated(atom: Atom) -> Atom {
        let mut atom = atom;
        match &mut atom {
            Atom::Expression(expr) => expr.set_evaluated(),
            _ => panic!("test helper only marks expressions as evaluated"),
        }
        atom
    }

    fn is_evaluated_expr(atom: &Atom) -> bool {
        matches!(atom, Atom::Expression(expr) if expr.is_evaluated())
    }

    #[derive(Clone, Debug)]
    enum GenAtom {
        Sym(u8),
        Var(u8),
        Expr(Vec<GenAtom>),
    }

    #[derive(Clone, Debug)]
    struct BindingOp {
        var: u8,
        value: GenAtom,
    }

    fn query_vars() -> Vec<VariableAtom> {
        vec![
            VariableAtom::new("x"),
            VariableAtom::new("y"),
            VariableAtom::new_id("v1_0", 10_001),
            VariableAtom::new_id("v1_1", 10_001),
        ]
    }

    fn gen_atom_to_atom(atom: &GenAtom, vars: &[VariableAtom]) -> Atom {
        match atom {
            GenAtom::Sym(i) => Atom::sym(format!("s{i}")),
            GenAtom::Var(i) => Atom::Variable(vars[*i as usize % vars.len()].clone()),
            GenAtom::Expr(children) => Atom::expr(
                children
                    .iter()
                    .map(|child| gen_atom_to_atom(child, vars))
                    .collect::<Vec<_>>(),
            ),
        }
    }

    fn arb_atom() -> impl Strategy<Value = GenAtom> {
        let leaf = prop_oneof![
            (0u8..8).prop_map(GenAtom::Sym),
            (0u8..4).prop_map(GenAtom::Var)
        ];
        leaf.prop_recursive(4, 32, 4, |inner| {
            prop::collection::vec(inner, 1..=4).prop_map(GenAtom::Expr)
        })
    }

    fn arb_ground_atom() -> impl Strategy<Value = GenAtom> {
        (0u8..8)
            .prop_map(GenAtom::Sym)
            .prop_recursive(4, 32, 4, |inner| {
                prop::collection::vec(inner, 1..=4).prop_map(GenAtom::Expr)
            })
    }

    fn arb_binding_op() -> impl Strategy<Value = BindingOp> {
        (0u8..4, arb_atom()).prop_map(|(var, value)| BindingOp { var, value })
    }

    fn canonical_atom_with_vars(atom: &Atom, vars: &mut HashMap<VariableAtom, usize>) -> String {
        match atom {
            Atom::Symbol(s) => format!("S({})", s.name()),
            Atom::Variable(v) => {
                let next = vars.len();
                let id = *vars.entry(v.clone()).or_insert(next);
                format!("V{id}")
            }
            Atom::Grounded(g) => format!("G({g})"),
            Atom::Expression(e) => {
                let children = e
                    .children()
                    .iter()
                    .map(|child| canonical_atom_with_vars(child, vars))
                    .collect::<Vec<_>>();
                format!("E({})", children.join(" "))
            }
        }
    }

    fn canonical_atom(atom: &Atom) -> String {
        canonical_atom_with_vars(atom, &mut HashMap::new())
    }

    fn query_variables(atom: &Atom) -> Vec<VariableAtom> {
        let mut seen = HashSet::new();
        let mut vars = Vec::new();
        for var in atom.iter().filter_type::<&VariableAtom>() {
            if seen.insert(var.clone()) {
                vars.push(var.clone());
            }
        }
        vars
    }

    fn projected_results(results: &BindingsSet, query_vars: &[VariableAtom]) -> Vec<Vec<String>> {
        let mut rows = results
            .iter()
            .map(|bindings| {
                let mut canonical_vars = HashMap::new();
                query_vars
                    .iter()
                    .map(|var| {
                        let value = bindings
                            .resolve(var)
                            .unwrap_or_else(|| Atom::Variable(var.clone()));
                        format!(
                            "{}={}",
                            var.name(),
                            canonical_atom_with_vars(&value, &mut canonical_vars)
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        rows.sort();
        rows
    }

    /// Every atom in the space, canonicalized, as a set (via visit).
    fn space_atom_set(space: &MorkSpace) -> BTreeSet<String> {
        let collected = RefCell::new(BTreeSet::new());
        let mut visitor = |atom: Cow<Atom>| {
            collected.borrow_mut().insert(canonical_atom(&atom));
        };
        space.visit(&mut visitor).unwrap();
        collected.into_inner()
    }

    fn insert_unique_atoms(atoms: &[GenAtom]) -> (GroundingSpace, MorkSpace) {
        let vars = query_vars();
        let mut seen = BTreeSet::new();
        let mut ground = GroundingSpace::new();
        let mut mork = MorkSpace::new();
        for atom in atoms {
            let atom = gen_atom_to_atom(atom, &vars);
            if seen.insert(canonical_atom(&atom)) {
                ground.add(atom.clone());
                mork.add(atom);
            }
        }
        (ground, mork)
    }

    assert_impl_all!(MorkSnapshot: Send, Sync);
    // The vision assertion: one MorkSpace shared across threads -- queries are
    // &self, every cache is lock-based, the kernel's z3 table is locked, and
    // atoms themselves are Send + Sync on the thread-safe hyperon base.
    assert_impl_all!(MorkSpace: Send, Sync);

    #[test]
    fn query_side_evaluated_expression_survives_variable_capture() {
        let rule = Atom::expr([
            Atom::sym("="),
            Atom::expr([Atom::sym("id"), Atom::var("x")]),
            Atom::var("x"),
        ]);
        let query_value = evaluated(Atom::expr([Atom::sym("value"), Atom::sym("a")]));
        let query = Atom::expr([
            Atom::sym("="),
            Atom::expr([Atom::sym("id"), query_value]),
            Atom::var("out"),
        ]);
        let mut ground = GroundingSpace::new();
        let mut mork = MorkSpace::new();
        ground.add(rule.clone());
        mork.add(rule);

        let ground_results = ground.query(&query);
        let mork_results = mork.query(&query);
        assert_eq!(ground_results.len(), 1);
        assert_eq!(mork_results.len(), 1);

        let ground_out = resolve(ground_results.iter().next().unwrap(), "out").unwrap();
        let mork_out = resolve(mork_results.iter().next().unwrap(), "out").unwrap();
        assert!(is_evaluated_expr(&ground_out));
        assert!(is_evaluated_expr(&mork_out));
        assert_eq!(canonical_atom(&mork_out), canonical_atom(&ground_out));
    }

    /// The documented order boundary: for EQUAL-specificity facts, GroundingSpace
    /// returns insertion order and MorkSpace returns trie byte order -- a
    /// set-semantics trie has no insertion history, and a sidecar tracking one
    /// measured a 16x bulk-load regression for an ordering MeTTa leaves
    /// unspecified. The result SETS must agree, and MorkSpace's order must be
    /// deterministic (stable across repeated queries).
    #[test]
    fn equal_specificity_order_is_trie_order_and_sets_agree() {
        let first = Atom::expr([
            Atom::sym("="),
            Atom::expr([Atom::sym("choice"), Atom::var("x")]),
            Atom::expr([Atom::sym("z"), Atom::var("x")]),
        ]);
        let second = Atom::expr([
            Atom::sym("="),
            Atom::expr([Atom::sym("choice"), Atom::var("x")]),
            Atom::expr([Atom::sym("a"), Atom::var("x")]),
        ]);
        let query = Atom::expr([
            Atom::sym("="),
            Atom::expr([Atom::sym("choice"), Atom::sym("target")]),
            Atom::var("out"),
        ]);
        let mut ground = GroundingSpace::new();
        let mut mork = MorkSpace::new();
        for atom in [first, second] {
            ground.add(atom.clone());
            mork.add(atom);
        }

        let ground_results = ground
            .query(&query)
            .iter()
            .map(|bindings| canonical_atom(&resolve(bindings, "out").unwrap()))
            .collect::<Vec<_>>();
        let mork_results = mork
            .query(&query)
            .iter()
            .map(|bindings| canonical_atom(&resolve(bindings, "out").unwrap()))
            .collect::<Vec<_>>();

        // Grounding: insertion order. Mork: trie order ("a" before "z").
        assert_eq!(ground_results, vec!["E(S(z) S(target))", "E(S(a) S(target))"]);
        assert_eq!(mork_results, vec!["E(S(a) S(target))", "E(S(z) S(target))"]);
        let sorted = |mut v: Vec<String>| {
            v.sort();
            v
        };
        assert_eq!(sorted(mork_results.clone()), sorted(ground_results));
        let mork_again = mork
            .query(&query)
            .iter()
            .map(|bindings| canonical_atom(&resolve(bindings, "out").unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(mork_again, mork_results, "mork order must be deterministic");
    }

    #[test]
    fn query_results_follow_grounding_specificity_order() {
        let generic = Atom::expr([
            Atom::sym("="),
            Atom::expr([Atom::sym("bce"), Atom::var("env"), Atom::var("depth"), Atom::expr([Atom::sym(":"), Atom::var("proof"), Atom::var("theorem")])]),
            Atom::sym("generic"),
        ]);
        let modus_ponens = Atom::expr([
            Atom::sym("="),
            Atom::expr([Atom::sym("bce"), Atom::var("env"), Atom::expr([Atom::sym("S"), Atom::var("k")]), Atom::expr([Atom::sym(":"), Atom::expr([Atom::sym("ax-mp"), Atom::var("left"), Atom::var("right")]), Atom::var("theorem")])]),
            Atom::sym("modus-ponens"),
        ]);
        let axiom = Atom::expr([
            Atom::sym("="),
            Atom::expr([Atom::sym("bce"), Atom::var("env"), Atom::var("depth"), Atom::expr([Atom::sym(":"), Atom::sym("ax-3"), Atom::var("theorem")])]),
            Atom::sym("axiom"),
        ]);
        let mut ground = GroundingSpace::new();
        let mut mork = MorkSpace::new();
        for atom in [generic, modus_ponens, axiom] {
            ground.add(atom.clone());
            mork.add(atom);
        }

        let mp_query = Atom::expr([
            Atom::sym("="),
            Atom::expr([Atom::sym("bce"), Atom::sym("env"), Atom::expr([Atom::sym("S"), Atom::sym("Z")]), Atom::expr([Atom::sym(":"), Atom::var("proof"), Atom::sym("goal")])]),
            Atom::var("out"),
        ]);
        let ax_query = Atom::expr([
            Atom::sym("="),
            Atom::expr([Atom::sym("bce"), Atom::sym("env"), Atom::sym("Z"), Atom::expr([Atom::sym(":"), Atom::var("proof"), Atom::sym("goal")])]),
            Atom::var("out"),
        ]);
        let ordered = |space: &dyn Space, query: &Atom| {
            space
                .query(query)
                .iter()
                .map(|bindings| canonical_atom(&resolve(bindings, "out").unwrap()))
                .collect::<Vec<_>>()
        };

        assert_eq!(ordered(&mork, &mp_query), ordered(&ground, &mp_query));
        assert_eq!(ordered(&mork, &ax_query), ordered(&ground, &ax_query));
    }

    #[test]
    fn query_results_are_narrowed_to_query_variables() {
        let fact = Atom::expr([
            Atom::sym(":"),
            Atom::sym("ax-1"),
            Atom::expr([
                Atom::sym("->"),
                Atom::var("p"),
                Atom::expr([Atom::sym("->"), Atom::var("q"), Atom::var("p")]),
            ]),
        ]);
        let query = Atom::expr([
            Atom::sym(":"),
            Atom::var("label"),
            Atom::expr([
                Atom::sym("->"),
                Atom::var("outer"),
                Atom::expr([Atom::sym("->"), Atom::var("inner"), Atom::var("outer")]),
            ]),
        ]);
        let mut ground = GroundingSpace::new();
        let mut mork = MorkSpace::new();
        ground.add(fact.clone());
        mork.add(fact);

        let ground_results = ground.query(&query);
        let mork_results = mork.query(&query);
        let query_vars = query_variables(&query);
        assert_eq!(
            projected_results(&mork_results, &query_vars),
            projected_results(&ground_results, &query_vars),
        );

        let public_vars = mork_results
            .iter()
            .flat_map(|bindings| bindings.vars())
            .map(|var| var.name())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            public_vars,
            ["inner".to_string(), "label".to_string(), "outer".to_string()]
                .into_iter()
                .collect(),
        );
    }

    proptest! {
        #[test]
        fn binding_construction_never_emits_merge_panic(ops in prop::collection::vec(arb_binding_op(), 0..32)) {
            let vars = query_vars();
            let mut set = BindingsSet::single();
            for op in ops {
                let var = vars[op.var as usize % vars.len()].clone();
                let value = gen_atom_to_atom(&op.value, &vars);
                set = bind_or_equate(set, var, value);
            }

            for bindings in set.iter() {
                let merge_result = catch_unwind(AssertUnwindSafe(|| Bindings::new().merge(bindings)));
                prop_assert!(merge_result.is_ok());
                let iter_result = catch_unwind(AssertUnwindSafe(|| bindings.iter().collect::<Vec<_>>()));
                prop_assert!(iter_result.is_ok());
            }
        }

        #[test]
        fn codec_round_trips_symbols_exprs_and_variable_coreference(atom in arb_atom()) {
            let vars = query_vars();
            let atom = gen_atom_to_atom(&atom, &vars);
            let mut encoded_vars = Vec::new();
            let mut bytes = Vec::new();
            let mut sink = GroundedSink::ValueOnly;
            prop_assume!(encode_atom(&atom, &mut encoded_vars, &mut bytes, &mut sink).is_ok());

            let mut pos = 0usize;
            let mut ctx = DecodeCtx {
                ns: 1,
                var_counter: 0,
                grounded: None,
                query_vars: &[],
                evaluated: None,
                result_id: next_variable_id(),
            };
            let decoded = decode_atom(&bytes, &mut pos, &mut ctx);

            prop_assert_eq!(pos, bytes.len());
            prop_assert_eq!(decoded.as_ref().map(canonical_atom), Some(canonical_atom(&atom)));
        }

        #[test]
        fn query_matches_grounding_space_on_small_ground_spaces(
            atoms in prop::collection::vec(arb_ground_atom(), 0..12),
            query in arb_atom(),
        ) {
            let vars = query_vars();
            let query = gen_atom_to_atom(&query, &vars);
            let query_vars = query_variables(&query);
            let (ground, mork) = insert_unique_atoms(&atoms);

            let ground_results = ground.query(&query);
            let mork_results = mork.query(&query);

            prop_assert_eq!(
                projected_results(&mork_results, &query_vars),
                projected_results(&ground_results, &query_vars)
            );
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq, Hash)]
    struct StateCellModel {
        depth: u8,
        live: [u8; 2],
        query: u8,
    }

    impl StateCellModel {
        fn live_matches(&self) -> Vec<usize> {
            self.live
                .iter()
                .enumerate()
                .filter_map(|(idx, value)| (*value == self.query).then_some(idx))
                .collect()
        }

        fn reference_matches(&self) -> Vec<usize> {
            (0..self.live.len())
                .filter(|idx| self.live[*idx] == self.query)
                .collect()
        }
    }

    #[derive(Clone, Debug, Eq, PartialEq, Hash)]
    enum StateCellAction {
        SetCell { id: usize, value: u8 },
        SetQuery(u8),
    }

    struct StateCellSystem;

    impl stateright::Model for StateCellSystem {
        type State = StateCellModel;
        type Action = StateCellAction;

        fn init_states(&self) -> Vec<Self::State> {
            vec![StateCellModel {
                depth: 0,
                live: [0, 1],
                query: 0,
            }]
        }

        fn actions(&self, state: &Self::State, actions: &mut Vec<Self::Action>) {
            if state.depth >= 4 {
                return;
            }
            for id in 0..state.live.len() {
                for value in 0..=1 {
                    actions.push(StateCellAction::SetCell { id, value });
                }
            }
            for value in 0..=1 {
                actions.push(StateCellAction::SetQuery(value));
            }
        }

        fn next_state(
            &self,
            last_state: &Self::State,
            action: Self::Action,
        ) -> Option<Self::State> {
            let mut state = last_state.clone();
            state.depth += 1;
            match action {
                StateCellAction::SetCell { id, value } => state.live[id] = value,
                StateCellAction::SetQuery(value) => state.query = value,
            }
            Some(state)
        }

        fn properties(&self) -> Vec<stateright::Property<Self>> {
            vec![stateright::Property::<Self>::always(
                "live-value match equals reference",
                |_, state| state.live_matches() == state.reference_matches(),
            )]
        }
    }

    #[test]
    fn stateright_state_cell_model_checks_live_value_matching() {
        use stateright::{Checker, Model};

        let checker = StateCellSystem.checker().spawn_bfs().join();
        checker.assert_properties();
    }

    /// Two cells holding the same value stored in different atoms must stay distinct (identity,
    /// not display), mutation through one must be visible on query (shared cell), and a query
    /// holding a cell must match a stored cell by *current* value (the live-value post-filter).
    #[test]
    fn mutable_grounded_identity_and_live_value_match() {
        let mut space = MorkSpace::new();
        let a = MutCell::new(0);
        let b = MutCell::new(0); // same value as `a`, but a distinct cell
        space.add(Atom::expr([
            Atom::sym("box"),
            Atom::sym("A"),
            Atom::gnd(a.clone()),
        ]));
        space.add(Atom::expr([
            Atom::sym("box"),
            Atom::sym("B"),
            Atom::gnd(b.clone()),
        ]));

        // Identity: mutate A's cell to 7; B's stays 0 (no display-key collision).
        a.set(7);
        let qa = Atom::expr([Atom::sym("box"), Atom::sym("A"), Atom::var("x")]);
        let qb = Atom::expr([Atom::sym("box"), Atom::sym("B"), Atom::var("x")]);
        assert_eq!(
            resolve(space.query(&qa).iter().next().unwrap(), "x"),
            Some(Atom::gnd(MutCell::new(7)))
        );
        assert_eq!(
            resolve(space.query(&qb).iter().next().unwrap(), "x"),
            Some(Atom::gnd(MutCell::new(0)))
        );

        // Live-value match: a query cell holding 0 matches only B (A now holds 7).
        let probe = MutCell::new(0);
        let qmatch = Atom::expr([Atom::sym("box"), Atom::var("k"), Atom::gnd(probe.clone())]);
        let ks: Vec<Atom> = space
            .query(&qmatch)
            .iter()
            .filter_map(|bn| resolve(bn, "k"))
            .collect();
        assert_eq!(ks, vec![Atom::sym("B")]);

        // After mutating the probe to 7, the same query now matches only A.
        probe.set(7);
        let ks: Vec<Atom> = space
            .query(&qmatch)
            .iter()
            .filter_map(|bn| resolve(bn, "k"))
            .collect();
        assert_eq!(ks, vec![Atom::sym("A")]);
    }

    #[test]
    fn add_and_count() {
        let mut space = MorkSpace::new();
        assert_eq!(space.atom_count(), Some(0));
        assert_eq!(space.rejected_atom_count(), 0);
        space.add(parent("Tom", "Bob"));
        space.add(parent("Bob", "Ann"));
        assert_eq!(space.atom_count(), Some(2));
        assert_eq!(space.rejected_atom_count(), 0);
    }

    #[test]
    fn overlong_symbol_add_is_reported() {
        let mut space = MorkSpace::new();
        let overlong = Atom::sym("x".repeat(MAX_FIELD + 1));
        space.add(overlong);

        assert_eq!(space.atom_count(), Some(0));
        assert_eq!(space.rejected_atom_count(), 1);
    }

    #[test]
    fn sharded_overlong_symbol_add_is_reported() {
        let mut space = ShardedMorkSpace::new(4);
        let overlong = Atom::sym("x".repeat(MAX_FIELD + 1));

        assert!(!space.add(&overlong));
        assert_eq!(space.len(), 0);
        assert_eq!(space.rejected_atom_count(), 1);
    }

    #[test]
    fn round_trip_nested_atom() {
        let mut space = MorkSpace::new();
        let a = Atom::expr([
            Atom::sym("f"),
            Atom::expr([Atom::sym("g"), Atom::sym("a")]),
            Atom::sym("b"),
        ]);
        space.add(a.clone());
        let q = Atom::var("x");
        // (match $x) -- a bare variable matches every stored atom.
        let results = space.query(&q);
        let got: Vec<Atom> = results
            .iter()
            .filter_map(|b| b.resolve(&VariableAtom::new("x")))
            .collect();
        assert_eq!(got, vec![a]);
    }

    #[test]
    fn query_binds_one_variable() {
        let mut space = MorkSpace::new();
        space.add(parent("Tom", "Bob"));
        space.add(parent("Bob", "Ann"));
        let q = Atom::expr([Atom::sym("parent"), Atom::sym("Tom"), Atom::var("child")]);
        let results = space.query(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(
            results
                .iter()
                .next()
                .unwrap()
                .resolve(&VariableAtom::new("child")),
            Some(Atom::sym("Bob"))
        );
    }

    #[test]
    fn query_two_variables_multiple_results() {
        let mut space = MorkSpace::new();
        space.add(parent("Tom", "Bob"));
        space.add(parent("Bob", "Ann"));
        let q = Atom::expr([Atom::sym("parent"), Atom::var("p"), Atom::var("c")]);
        let results = space.query(&q);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn dynspace_wraps_mork_space_for_interpreter_use() {
        let space = hyperon_space::DynSpace::new(MorkSpace::new());
        assert_eq!(space.borrow().atom_count(), Some(0));
    }

    #[test]
    fn remove_atom() {
        let mut space = MorkSpace::new();
        space.add(parent("Tom", "Bob"));
        assert_eq!(space.len(), 1);
        assert!(space.remove(&parent("Tom", "Bob")));
        assert_eq!(space.len(), 0);
        assert!(!space.remove(&parent("Tom", "Bob")));
    }

    #[test]
    fn bulk_load_scales_past_the_trie_crash() {
        let mut space = MorkSpace::new();
        let mut text = String::new();
        for i in 0..20_000u32 {
            text.push_str(&format!("(edge n{} n{})\n", i, i + 1));
        }
        space.add_sexpr_text(&text).unwrap();
        assert_eq!(space.atom_count(), Some(20_000));
        let q = Atom::expr([Atom::sym("edge"), Atom::sym("n10000"), Atom::var("dst")]);
        assert_eq!(space.query(&q).len(), 1);
    }

    #[test]
    fn sharded_parallel_sweep_matches_sequential() {
        let mut space = ShardedMorkSpace::new(8);
        for i in 0..1000u32 {
            space.add(&Atom::expr([
                Atom::sym("edge"),
                Atom::sym(format!("a{}", i)),
                Atom::sym(format!("b{}", i)),
            ]));
        }
        assert_eq!(space.len(), 1000);
        let all = Atom::expr([Atom::sym("edge"), Atom::var("x"), Atom::var("y")]);
        assert_eq!(space.count_matches(&all), 1000);
        assert_eq!(space.par_count_matches(&all), 1000);
        // A partial pattern still sweeps every shard and finds the one match.
        let one = Atom::expr([Atom::sym("edge"), Atom::sym("a500"), Atom::var("y")]);
        assert_eq!(space.par_count_matches(&one), 1);
    }

    #[test]
    fn mm2_exec_runs_transitive_closure() {
        // Load facts + a forward-chaining exec rule, run MORK's MM2 exec to fixpoint,
        // then read the transitive closure back through the byte-level query.
        let mut space = MorkSpace::new();
        space
            .add_sexpr_text(
                "(edge a b)\n(edge b c)\n(edge c d)\n\
                 (path a b)\n(path b c)\n(path c d)\n\
                 (exec 0 (, (edge $x $y) (path $y $z)) (, (path $x $z)))\n",
            )
            .unwrap();
        // The plain exec rule is one-shot, so re-add it each round and step until the
        // closure stops growing (a tiny driver; semi-naive fixpoint is feature-gated).
        let mut prev = 0;
        for _ in 0..8 {
            space
                .add_sexpr_text("(exec 0 (, (edge $x $y) (path $y $z)) (, (path $x $z)))\n")
                .unwrap();
            space.step(1);
            let now = space.len();
            if now == prev {
                break;
            }
            prev = now;
        }
        let q = Atom::expr([Atom::sym("path"), Atom::sym("a"), Atom::var("z")]);
        let mut zs: Vec<String> = space
            .query(&q)
            .iter()
            .filter_map(|b| b.resolve(&VariableAtom::new("z")))
            .map(|a| a.to_string())
            .collect();
        zs.sort();
        zs.dedup();
        assert_eq!(zs, vec!["b", "c", "d"]); // transitive closure from a: b, c, d
    }

    /// hyperon-experimental #1079: GroundingSpace::visit undercounts atoms when many
    /// share a head symbol (its reproducer enumerated 1293 of 1500). MORK walks every
    /// trie leaf, so visit / get-atoms / atom_count are exact. Same reproducer workload.
    #[test]
    fn visit_counts_all_same_head_atoms() {
        struct Counter(usize);
        impl SpaceVisitor for Counter {
            fn accept(&mut self, _: Cow<'_, Atom>) {
                self.0 += 1;
            }
        }
        let mut space = MorkSpace::new();
        let target = 1500usize;
        for n in 0..target {
            space.add(Atom::expr([
                Atom::sym("item-shape-signal"),
                Atom::sym(format!("pattern-{n}")),
                Atom::sym(format!("target-shape-{n}")),
                Atom::sym(format!("{n}")),
            ]));
        }
        assert_eq!(space.atom_count(), Some(target));
        let mut counter = Counter(0);
        space.visit(&mut counter).unwrap();
        assert_eq!(
            counter.0, target,
            "MORK visit enumerated {} of {}",
            counter.0, target
        );
    }

    /// The Space trait's own doc example: `(, (A $x) ($x C))` over {(A B), (B C)}
    /// joins on $x natively (one kernel multi-factor query, not a per-conjunct loop).
    #[test]
    fn conjunctive_query_joins_on_the_shared_variable() {
        let atoms = vec![
            Atom::expr([Atom::sym("A"), Atom::sym("B")]),
            Atom::expr([Atom::sym("B"), Atom::sym("C")]),
        ];
        let mut ground = GroundingSpace::new();
        let mut mork = MorkSpace::new();
        for a in &atoms {
            ground.add(a.clone());
            mork.add(a.clone());
        }
        let query = Atom::expr([
            Atom::sym(","),
            Atom::expr([Atom::sym("A"), Atom::var("x")]),
            Atom::expr([Atom::var("x"), Atom::sym("C")]),
        ]);
        let vars = query_variables(&query);
        assert_eq!(
            projected_results(&mork.query(&query), &vars),
            projected_results(&ground.query(&query), &vars),
        );
        // And concretely: exactly one result, x = B.
        let results = mork.query(&query);
        assert_eq!(results.len(), 1);
        assert_eq!(
            results.iter().next().unwrap().resolve(&VariableAtom::new("x")),
            Some(Atom::sym("B"))
        );
    }

    /// A conjunction with no shared variable is the cross product, same as
    /// GroundingSpace's fold; the empty conjunction is one empty binding.
    #[test]
    fn conjunction_edge_shapes_match_grounding_space() {
        let atoms = vec![
            Atom::expr([Atom::sym("p"), Atom::sym("1")]),
            Atom::expr([Atom::sym("p"), Atom::sym("2")]),
            Atom::expr([Atom::sym("q"), Atom::sym("3")]),
        ];
        let mut ground = GroundingSpace::new();
        let mut mork = MorkSpace::new();
        for a in &atoms {
            ground.add(a.clone());
            mork.add(a.clone());
        }
        for query in [
            // cross product: 2 x 1 results
            Atom::expr([
                Atom::sym(","),
                Atom::expr([Atom::sym("p"), Atom::var("a")]),
                Atom::expr([Atom::sym("q"), Atom::var("b")]),
            ]),
            // single conjunct == plain query
            Atom::expr([Atom::sym(","), Atom::expr([Atom::sym("p"), Atom::var("a")])]),
            // empty conjunction
            Atom::expr([Atom::sym(",")]),
            // three-way join with a chained variable
            Atom::expr([
                Atom::sym(","),
                Atom::expr([Atom::sym("p"), Atom::var("a")]),
                Atom::expr([Atom::sym("p"), Atom::var("b")]),
                Atom::expr([Atom::sym("q"), Atom::var("c")]),
            ]),
        ] {
            let vars = query_variables(&query);
            assert_eq!(
                projected_results(&mork.query(&query), &vars),
                projected_results(&ground.query(&query), &vars),
                "diverged on {query}",
            );
        }
    }

    proptest! {
        /// Randomized conjunctions over the shared variable pool: the native
        /// multi-factor join returns exactly GroundingSpace's fold semantics,
        /// including cross-conjunct variable joins and products.
        #[test]
        fn conjunctive_query_matches_grounding_space(
            atoms in prop::collection::vec(arb_ground_atom(), 0..12),
            conj in prop::collection::vec(arb_atom(), 1..4),
        ) {
            let vars = query_vars();
            let mut children = vec![Atom::sym(",")];
            children.extend(conj.iter().map(|c| gen_atom_to_atom(c, &vars)));
            let query = Atom::expr(children);
            let qv = query_variables(&query);
            let (ground, mork) = insert_unique_atoms(&atoms);

            prop_assert_eq!(
                projected_results(&mork.query(&query), &qv),
                projected_results(&ground.query(&query), &qv)
            );
        }
    }

    /// The counterexample that forced the var-freeness latch: a bare-variable
    /// fact unifies with any factor, but the factorized fold's prefix seek
    /// cannot see it (it counted 0 where the matcher counts 1). Pinned so the
    /// latch never silently loosens.
    #[cfg(feature = "factorized-aggregate")]
    #[test]
    fn schematic_store_stays_on_the_enumerating_count() {
        let mut space = MorkSpace::new();
        space.add(Atom::var("anything"));
        let query = Atom::expr([
            Atom::sym(","),
            Atom::expr([Atom::sym("s")]),
            Atom::expr([Atom::sym("s")]),
        ]);
        assert_eq!(space.snapshot().count_matches(&query), 1);
    }

    /// On a variable-free store the factorized path is admitted and must agree
    /// with the enumerating count exactly.
    #[cfg(feature = "factorized-aggregate")]
    proptest! {
        #[test]
        fn factorized_count_matches_the_enumerating_count_on_ground_stores(
            atoms in prop::collection::vec(arb_ground_atom(), 0..14),
            conj in prop::collection::vec(arb_atom(), 2..4),
        ) {
            let vars = query_vars();
            let mut mork = MorkSpace::new();
            for a in &atoms {
                mork.add(gen_atom_to_atom(a, &vars));
            }
            let mut children = vec![Atom::sym(",")];
            children.extend(conj.iter().map(|c| gen_atom_to_atom(c, &vars)));
            let query = Atom::expr(children);

            let snap = mork.snapshot();
            let routed = snap.count_matches(&query);

            let reference = match wrap_pattern(&query) {
                None => 0,
                Some(mut pattern) => {
                    let pat_expr = Expr { ptr: pattern.bytes.as_mut_ptr() };
                    MorkKernel::query_multi(&snap.btm, pat_expr, |_res, _loc| true)
                }
            };

            prop_assert_eq!(routed, reference);
        }
    }

    /// The routed query (column index and all) must equal the raw matcher
    /// query on every query shape over a ground store: routing may change the
    /// complexity, never the answers.
    proptest! {
        #[test]
        fn routed_query_equals_the_matcher_on_ground_stores(
            atoms in prop::collection::vec(arb_ground_atom(), 0..14),
            query in arb_atom(),
        ) {
            let vars = query_vars();
            let query = gen_atom_to_atom(&query, &vars);
            let qv = query_variables(&query);
            let mut mork = MorkSpace::new();
            let mut reg = GroundedRegistry::default();
            for a in &atoms {
                let a = gen_atom_to_atom(a, &vars);
                reg.register(&a);
                mork.add(a);
            }
            let routed = mork.query(&query);
            let reference = query_btm(&mork.kernel.btm, &query, Some(&reg));
            prop_assert_eq!(
                projected_results(&routed, &qv),
                projected_results(&reference, &qv)
            );
        }
    }

    /// reduce() runs evaluation as MM2 rewriting: accumulator-style recursion
    /// reaches its normal form, nondeterministic equations return every
    /// branch, an equation-free term self-returns, and the live space never
    /// sees the scaffolding.
    #[test]
    fn reduce_runs_metta_rewriting_on_the_kernel() {
        let mut space = MorkSpace::new();
        space
            .add_sexpr_text("(= (add Z $y) $y)
(= (add (S $x) $y) (add $x (S $y)))
")
            .unwrap();

        let two_plus_three = Atom::expr([
            Atom::sym("add"),
            peano_test_atom(2),
            peano_test_atom(3),
        ]);
        let results = space.reduce(&two_plus_three, 50);
        assert_eq!(results, vec![peano_test_atom(5)], "2+3 must normalize to 5");

        // Nondeterminism: both coin branches come back.
        space.add_sexpr_text("(= (coin) heads)
(= (coin) tails)
").unwrap();
        let mut coins = space
            .reduce(&Atom::expr([Atom::sym("coin")]), 10)
            .iter()
            .map(canonical_atom)
            .collect::<Vec<_>>();
        coins.sort();
        assert_eq!(coins, vec!["S(heads)".to_string(), "S(tails)".to_string()]);

        // No equation matches: the term self-returns.
        let inert = Atom::expr([Atom::sym("opaque"), Atom::sym("thing")]);
        assert_eq!(space.reduce(&inert, 10), vec![inert.clone()]);

        // The live space holds only the program: no mm2-ev, no exec leftovers.
        assert!(space
            .query(&Atom::expr([Atom::sym("mm2-ev"), Atom::var("t")]))
            .is_empty());
    }

    fn peano_test_atom(n: usize) -> Atom {
        let mut a = Atom::sym("Z");
        for _ in 0..n {
            a = Atom::expr([Atom::sym("S"), a]);
        }
        a
    }

    /// The auto-tabler admits by measured cost: the admission predicate is
    /// checked with explicit costs (no wall clock), and the integration half
    /// pins the scan-class behavior, a 30k-store scan tabling and replaying
    /// exactly. A small store's fill is deliberately not asserted untabled end
    /// to end: the gate adapts to real elapsed time, so on a downclocked or
    /// loaded core a cold 1-atom fill can legitimately cross the threshold.
    #[test]
    fn tabling_admits_only_costly_queries() {
        assert!(
            !worth_tabling(std::time::Duration::from_micros(5), 1),
            "a microsecond fill must not be tabled"
        );
        assert!(
            worth_tabling(QUERY_TABLE_MIN_COST, 1),
            "a threshold-cost fill must be tabled"
        );
        assert!(
            !worth_tabling(std::time::Duration::from_secs(1), QUERY_CACHE_MAX_ROWS + 1),
            "an oversized result set must not be tabled at any cost"
        );

        let needle_query = Atom::expr([Atom::var("x"), Atom::sym("mid"), Atom::var("y")]);

        let mut small = MorkSpace::new();
        small.add(Atom::expr([Atom::sym("g"), Atom::sym("mid"), Atom::sym("c")]));
        assert_eq!(small.query(&needle_query).len(), 1);

        let mut big = MorkSpace::new();
        for i in 0..30_000u32 {
            big.add(Atom::expr([
                Atom::sym(format!("f{i}")),
                Atom::sym(format!("a{i}")),
                Atom::sym(format!("b{i}")),
            ]));
        }
        big.add(Atom::expr([Atom::sym("g"), Atom::sym("mid"), Atom::sym("c")]));
        let qv = query_variables(&needle_query);
        let first = big.query(&needle_query);
        assert_eq!(
            big.query_cache.read().unwrap().len(),
            1,
            "a scan-class fill must be tabled"
        );
        let replay = big.query(&needle_query);
        let fresh = query_btm(&big.kernel.btm, &needle_query, Some(&big.grounded));
        assert_eq!(projected_results(&first, &qv), projected_results(&fresh, &qv));
        assert_eq!(projected_results(&replay, &qv), projected_results(&fresh, &qv));
    }

    /// transform() must equal querying the conjunction and adding every
    /// instantiated template -- the reference semantics of the wiki's
    /// Transform operation.
    #[test]
    fn transform_materializes_the_join_directly() {
        let mut space = MorkSpace::new();
        for (a, b) in [("a", "b"), ("b", "c"), ("c", "d")] {
            space.add(Atom::expr([Atom::sym("edge"), Atom::sym(a), Atom::sym(b)]));
        }
        let patterns = [
            Atom::expr([Atom::sym("edge"), Atom::var("x"), Atom::var("m")]),
            Atom::expr([Atom::sym("edge"), Atom::var("m"), Atom::var("y")]),
        ];
        let templates = [Atom::expr([Atom::sym("path"), Atom::var("x"), Atom::var("y")])];
        let (matches, any_new) = space.transform(&patterns, &templates).unwrap();
        assert_eq!(matches, 2, "a 4-node chain has two 2-hop pairs");
        assert!(any_new);
        let paths = space.query(&Atom::expr([
            Atom::sym("path"),
            Atom::var("p"),
            Atom::var("q"),
        ]));
        assert_eq!(paths.len(), 2);
        // Idempotent re-run: same matches, nothing new.
        let (matches2, any_new2) = space.transform(&patterns, &templates).unwrap();
        assert_eq!(matches2, 2);
        assert!(!any_new2);
    }

    proptest! {
        /// Randomized transform differential against query-then-add.
        #[test]
        fn transform_equals_query_then_add(
            atoms in prop::collection::vec(arb_ground_atom(), 0..10),
            pats in prop::collection::vec(arb_atom(), 1..3),
            tpls in prop::collection::vec(arb_atom(), 1..3),
        ) {
            let vars = query_vars();
            let patterns: Vec<Atom> = pats.iter().map(|a| gen_atom_to_atom(a, &vars)).collect();
            let templates: Vec<Atom> = tpls.iter().map(|a| gen_atom_to_atom(a, &vars)).collect();
            // The kernel emits a fresh variable for a template var unbound by the
            // patterns; the reference would keep the caller's name. Exclude that
            // shape: template vars must all occur in the patterns.
            let pattern_vars: HashSet<VariableAtom> = patterns
                .iter()
                .flat_map(|p| p.iter().filter_type::<&VariableAtom>().cloned())
                .collect();
            for t in &templates {
                for v in t.iter().filter_type::<&VariableAtom>() {
                    prop_assume!(pattern_vars.contains(v));
                }
            }
            let mut base = MorkSpace::new();
            for a in &atoms {
                base.add(gen_atom_to_atom(a, &vars));
            }

            let mut direct = base.fork();
            let transformed = direct.transform(&patterns, &templates);
            prop_assume!(transformed.is_some());
            let (matches, _any_new) = transformed.unwrap();

            let mut reference = base.fork();
            let mut conj = vec![Atom::sym(",")];
            conj.extend(patterns.iter().cloned());
            let rows = reference.query(&Atom::expr(conj));
            prop_assert_eq!(matches, rows.len(), "match count vs join rows");
            for b in rows.iter() {
                for t in &templates {
                    let instantiated =
                        hyperon_atom::matcher::apply_bindings_to_atom_move(t.clone(), b);
                    reference.add(instantiated);
                }
            }
            prop_assert_eq!(space_atom_set(&direct), space_atom_set(&reference));
        }
    }

    /// Restriction keeps exactly the subtree under the pattern's ground prefix.
    #[test]
    fn restriction_is_a_prefix_gate() {
        let mut space = MorkSpace::new();
        for atom in [
            Atom::expr([Atom::sym("rel"), Atom::sym("a"), Atom::sym("1")]),
            Atom::expr([Atom::sym("rel"), Atom::sym("a"), Atom::sym("2")]),
            Atom::expr([Atom::sym("rel"), Atom::sym("b"), Atom::sym("3")]),
            Atom::expr([Atom::sym("other"), Atom::sym("a"), Atom::sym("9")]),
        ] {
            space.add(atom);
        }
        // Everything survives a bare-variable gate (empty ground prefix).
        let mut all = space.fork();
        assert_eq!(all.restrict_to_prefix(&Atom::var("x")), Some(4));

        let kept = space
            .restrict_to_prefix(&Atom::expr([
                Atom::sym("rel"),
                Atom::sym("a"),
                Atom::var("x"),
            ]))
            .unwrap();
        assert_eq!(kept, 2);
        assert_eq!(
            space
                .query(&Atom::expr([Atom::sym("rel"), Atom::sym("a"), Atom::var("v")]))
                .len(),
            2
        );
        assert!(space
            .query(&Atom::expr([Atom::sym("other"), Atom::var("a"), Atom::var("b")]))
            .is_empty());
    }

    /// save_paths/load_paths round-trips the trie through disk.
    #[test]
    fn paths_roundtrip_through_disk() {
        let mut space = MorkSpace::new();
        space.add(Atom::expr([Atom::sym("fact"), Atom::sym("x")]));
        space.add(Atom::expr([Atom::sym("rule"), Atom::var("v")]));
        let file = std::env::temp_dir().join(format!(
            "metta-on-mork-paths-{}.bin",
            std::process::id()
        ));
        space.save_paths(&file).unwrap();
        let mut restored = MorkSpace::new();
        restored.load_paths(&file).unwrap();
        let _ = std::fs::remove_file(&file);
        assert_eq!(space_atom_set(&restored), space_atom_set(&space));
    }

    /// Parallel bulk load must land exactly the atoms a sequential add loop
    /// lands: same atom set, same rejection count, same latch.
    proptest! {
        #[test]
        fn parallel_load_equals_sequential(
            atoms in prop::collection::vec(arb_atom(), 0..24),
        ) {
            let vars = query_vars();
            let atoms: Vec<Atom> = atoms.iter().map(|a| gen_atom_to_atom(a, &vars)).collect();
            let mut seq = MorkSpace::new();
            for a in &atoms {
                seq.add(a.clone());
            }
            let mut par = MorkSpace::new();
            par.extend_parallel(&atoms, 4);
            prop_assert_eq!(space_atom_set(&par), space_atom_set(&seq));
            prop_assert_eq!(par.rejected_atom_count(), seq.rejected_atom_count());
            prop_assert_eq!(par.var_free, seq.var_free);
        }
    }

    /// Mutable grounded atoms take the sequential fallback inside the parallel
    /// load and keep their identity semantics.
    #[test]
    fn parallel_load_handles_mutable_grounded() {
        let cell = MutCell::new(3);
        let atoms = vec![
            Atom::expr([Atom::sym("box"), Atom::gnd(cell.clone())]),
            Atom::expr([Atom::sym("plain"), Atom::sym("x")]),
        ];
        let mut space = MorkSpace::new();
        space.extend_parallel(&atoms, 2);
        cell.set(9);
        let q = Atom::expr([Atom::sym("box"), Atom::var("v")]);
        assert_eq!(
            resolve(space.query(&q).iter().next().unwrap(), "v"),
            Some(Atom::gnd(MutCell::new(9))),
            "live value must be visible through the registry"
        );
    }

    /// fork() is copy-on-write isolation: the fork answers like the original
    /// until either side mutates, and mutations never leak across.
    #[test]
    fn fork_isolates_mutations_both_ways() {
        let mut original = MorkSpace::new();
        original.add(Atom::expr([Atom::sym("p"), Atom::sym("1")]));
        let mut fork = original.fork();
        let q = Atom::expr([Atom::sym("p"), Atom::var("x")]);
        assert_eq!(fork.query(&q).len(), 1);
        fork.add(Atom::expr([Atom::sym("p"), Atom::sym("2")]));
        original.add(Atom::expr([Atom::sym("p"), Atom::sym("3")]));
        assert_eq!(fork.query(&q).len(), 2);
        assert_eq!(original.query(&q).len(), 2);
        assert!(fork.query(&Atom::expr([Atom::sym("p"), Atom::sym("3")])).is_empty());
        assert!(original.query(&Atom::expr([Atom::sym("p"), Atom::sym("2")])).is_empty());
    }

    /// The trie algebra must equal the per-atom set semantics on ground spaces,
    /// and decline (leaving self untouched) when mutable grounded atoms exist.
    proptest! {
        #[test]
        fn trie_algebra_matches_per_atom_set_semantics(
            a in prop::collection::vec(arb_ground_atom(), 0..12),
            b in prop::collection::vec(arb_ground_atom(), 0..12),
        ) {
            let vars = query_vars();
            let build = |atoms: &[GenAtom]| {
                let mut space = MorkSpace::new();
                let mut set = BTreeSet::new();
                for g in atoms {
                    let atom = gen_atom_to_atom(g, &vars);
                    set.insert(canonical_atom(&atom));
                    space.add(atom);
                }
                (space, set)
            };
            let atoms_of = space_atom_set;

            let (a_space, a_set) = build(&a);
            let (b_space, b_set) = build(&b);

            let mut u = a_space.fork();
            prop_assert!(u.union_with(&b_space));
            prop_assert_eq!(atoms_of(&u), a_set.union(&b_set).cloned().collect::<BTreeSet<_>>());

            let mut i = a_space.fork();
            prop_assert!(i.intersect_with(&b_space));
            prop_assert_eq!(atoms_of(&i), a_set.intersection(&b_set).cloned().collect::<BTreeSet<_>>());

            let mut d = a_space.fork();
            prop_assert!(d.subtract_space(&b_space));
            prop_assert_eq!(atoms_of(&d), a_set.difference(&b_set).cloned().collect::<BTreeSet<_>>());
        }
    }

    /// Mutable grounded atoms have space-local identity ids, so algebra with
    /// them must decline rather than merge dangling id bytes.
    #[test]
    fn algebra_declines_mutable_grounded_stores() {
        let mut a = MorkSpace::new();
        a.add(Atom::expr([Atom::sym("cell"), Atom::gnd(MutCell::new(1))]));
        let b = MorkSpace::new();
        let before = a.atom_count();
        assert!(!a.union_with(&b));
        let mut c = MorkSpace::new();
        c.add(Atom::sym("plain"));
        assert!(!c.union_with(&a), "other side's mutable atoms must also decline");
        assert_eq!(a.atom_count(), before);
    }

    /// Tabled replays must be indistinguishable from live matches: the same
    /// query repeated (replay path), and repeated again after a mutation
    /// (invalidation), each equal a fresh matcher run -- over stores that may
    /// hold variables (the cache does not need the var-freeness latch).
    proptest! {
        #[test]
        fn tabled_queries_replay_exactly(
            atoms in prop::collection::vec(arb_atom(), 0..14),
            extra in arb_ground_atom(),
            query in arb_atom(),
        ) {
            let vars = query_vars();
            let query = gen_atom_to_atom(&query, &vars);
            let qv = query_variables(&query);
            let mut mork = MorkSpace::new();
            let mut reg = GroundedRegistry::default();
            for a in &atoms {
                let a = gen_atom_to_atom(a, &vars);
                reg.register(&a);
                mork.add(a);
            }
            let fresh = |m: &MorkSpace, r: &GroundedRegistry| {
                query_btm(&m.kernel.btm, &query, Some(r))
            };
            let first = mork.query(&query);
            let replay = mork.query(&query);
            prop_assert_eq!(
                projected_results(&replay, &qv),
                projected_results(&fresh(&mork, &reg), &qv)
            );
            prop_assert_eq!(
                projected_results(&first, &qv),
                projected_results(&replay, &qv)
            );
            let extra = gen_atom_to_atom(&extra, &vars);
            reg.register(&extra);
            mork.add(extra);
            let after = mork.query(&query);
            prop_assert_eq!(
                projected_results(&after, &qv),
                projected_results(&fresh(&mork, &reg), &qv)
            );
        }
    }

    /// A snapshot's carried (frozen) indexes must answer exactly like the
    /// matcher over the same trie, for every query shape.
    proptest! {
        #[test]
        fn snapshot_carried_indexes_equal_the_matcher(
            atoms in prop::collection::vec(arb_ground_atom(), 0..14),
            query in arb_atom(),
        ) {
            let vars = query_vars();
            let query = gen_atom_to_atom(&query, &vars);
            let qv = query_variables(&query);
            let mut mork = MorkSpace::new();
            for a in &atoms {
                mork.add(gen_atom_to_atom(a, &vars));
            }
            // Warm the live space's index for this shape (if the route admits
            // it), then snapshot: the snapshot carries the frozen index.
            let _ = mork.query(&query);
            let snap = mork.snapshot();
            prop_assert_eq!(
                projected_results(&snap.query(&query), &qv),
                projected_results(&query_btm(&snap.btm, &query, None), &qv)
            );
        }
    }

    /// The generation counter really invalidates: a fact added after the index
    /// was built must appear in the next indexed answer.
    #[test]
    fn column_index_rebuilds_after_mutation() {
        let mut space = MorkSpace::new();
        space.add(Atom::expr([Atom::sym("edge"), Atom::sym("a"), Atom::sym("t")]));
        let q = Atom::expr([Atom::sym("edge"), Atom::var("x"), Atom::sym("t")]);
        assert_eq!(space.query(&q).len(), 1);
        space.add(Atom::expr([Atom::sym("edge"), Atom::sym("b"), Atom::sym("t")]));
        assert_eq!(space.query(&q).len(), 2, "stale index after add");
        space.remove(&Atom::expr([Atom::sym("edge"), Atom::sym("a"), Atom::sym("t")]));
        assert_eq!(space.query(&q).len(), 1, "stale index after remove");
    }

    /// The schematic-store differential: variable-bearing stores must fall back
    /// to the enumerating count (the latch), keeping counts exact everywhere.
    #[cfg(feature = "factorized-aggregate")]
    proptest! {
        #[test]
        fn factorized_count_matches_the_enumerating_count(
            atoms in prop::collection::vec(arb_atom(), 0..14),
            conj in prop::collection::vec(arb_atom(), 2..4),
        ) {
            let vars = query_vars();
            let mut mork = MorkSpace::new();
            for a in &atoms {
                mork.add(gen_atom_to_atom(a, &vars));
            }
            let mut children = vec![Atom::sym(",")];
            children.extend(conj.iter().map(|c| gen_atom_to_atom(c, &vars)));
            let query = Atom::expr(children);

            let snap = mork.snapshot();
            let routed = snap.count_matches(&query);

            let reference = match wrap_pattern(&query) {
                None => 0,
                Some(mut pattern) => {
                    let pat_expr = Expr { ptr: pattern.bytes.as_mut_ptr() };
                    MorkKernel::query_multi(&snap.btm, pat_expr, |_res, _loc| true)
                }
            };

            prop_assert_eq!(routed, reference);
        }
    }

    // ===== Phase 1: visit_query tests =====

    /// Compatibility: visit_query count matches query().len() for small dataset.
    #[test]
    fn visit_query_count_matches_query_len() {
        let mut space = MorkSpace::new();
        for i in 0..100u32 {
            space.add(Atom::expr([Atom::sym("edge"), Atom::sym(format!("n{i}")), Atom::sym(format!("n{}", i + 1))]));
        }
        let query = Atom::expr([Atom::sym("edge"), Atom::var("from"), Atom::var("to")]);
        let query_len = space.query(&query).len();
        let visit_count = space.visit_query(&query, &mut |_| true);
        assert_eq!(query_len, visit_count);
    }

    /// Early termination: stop after 10 results, verify exactly 10 delivered.
    #[test]
    fn visit_query_early_stop() {
        let mut space = MorkSpace::new();
        for i in 0..1000u32 {
            space.add(Atom::expr([Atom::sym("edge"), Atom::sym(format!("n{i}")), Atom::sym(format!("n{}", i + 1))]));
        }
        let query = Atom::expr([Atom::sym("edge"), Atom::var("from"), Atom::var("to")]);
        let mut delivered = 0u32;
        let count = space.visit_query(&query, &mut |_| {
            delivered += 1;
            delivered < 10
        });
        assert_eq!(count, 10);
        assert_eq!(delivered, 10);
    }

    /// Streaming: large dataset, visit_query returns correct count, no panic.
    #[test]
    fn visit_query_large_dataset() {
        let mut space = MorkSpace::new();
        for i in 0..50_000u32 {
            space.add(Atom::expr([Atom::sym("fact"), Atom::sym(format!("item{i}"))]));
        }
        let query = Atom::expr([Atom::sym("fact"), Atom::var("x")]);
        let query_len = space.query(&query).len();
        let visit_count = space.visit_query(&query, &mut |_| true);
        assert_eq!(query_len, 50_000);
        assert_eq!(visit_count, 50_000);
    }

    /// Empty result set: visit_query returns 0.
    #[test]
    fn visit_query_empty() {
        let space = MorkSpace::new();
        let query = Atom::expr([Atom::sym("nope"), Atom::var("x")]);
        let count = space.visit_query(&query, &mut |_| true);
        assert_eq!(count, 0);
    }

    /// All bindings are valid: no panics, all bindings contain expected variables.
    #[test]
    fn visit_query_bindings_valid() {
        let mut space = MorkSpace::new();
        space.add(Atom::expr([Atom::sym("knows"), Atom::sym("Alice"), Atom::sym("Bob")]));
        space.add(Atom::expr([Atom::sym("knows"), Atom::sym("Alice"), Atom::sym("Charlie")]));
        let query = Atom::expr([Atom::sym("knows"), Atom::sym("Alice"), Atom::var("who")]);
        let mut names = Vec::new();
        space.visit_query(&query, &mut |b| {
            if let Some(val) = b.resolve(&VariableAtom::new("who")) {
                names.push(val.to_string());
            }
            true
        });
        names.sort();
        assert_eq!(names, vec!["Bob", "Charlie"]);
    }
}
