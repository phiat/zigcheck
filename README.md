# zigcheck

[![Zig](https://img.shields.io/badge/Zig-0.17.0-f7a41d?logo=zig&logoColor=white)](https://ziglang.org)
[![Version](https://img.shields.io/badge/version-0.6.1-orange)](build.zig.zon)
[![Tests](https://img.shields.io/badge/tests-187_passing-brightgreen)](#development)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

Property-based testing for Zig, inspired by Haskell's [QuickCheck](https://hackage.haskell.org/package/QuickCheck). Generate random structured inputs, check properties, and shrink failing cases to minimal counterexamples. Requires Zig 0.17.0 (use v0.5.5 for 0.15).

## Install

```zig
// build.zig.zon
.dependencies = .{
    .zigcheck = .{
        .url = "https://github.com/phiat/zigcheck/archive/refs/tags/v0.6.1.tar.gz",
        // run `zig build` once; the compiler prints the .hash to add
    },
},
```

```zig
// build.zig
const zigcheck_mod = b.dependency("zigcheck", .{}).module("zigcheck");
my_tests.root_module.addImport("zigcheck", zigcheck_mod);
```

## Quick start

The first property to reach for: **if you encode it, you can decode it back**.

```zig
const std = @import("std");
const zigcheck = @import("zigcheck");

test "integers survive format/parse roundtrip" {
    try zigcheck.forAll(i32, zigcheck.generators.int(i32), struct {
        fn prop(n: i32) !void {
            var buf: [20]u8 = undefined;
            const str = std.fmt.bufPrint(&buf, "{d}", .{n}) catch return;
            const parsed = std.fmt.parseInt(i32, str, 10) catch
                return error.PropertyFalsified;
            if (parsed != n) return error.PropertyFalsified;
        }
    }.prop);
}
```

Zig has no function literals, so every property is `struct { fn prop(...) ... }.prop`. That is the intended form.

When a property fails, zigcheck shrinks the input to a minimal counterexample:

```
--- zigcheck: FAILED after 12 tests ------------------------
  Counterexample: 10
  Shrunk (8 steps) from: 1847382901
  Reproduction seed: 0x2a
  Rerun with: .seed = 0x2a
-------------------------------------------------------------
```

### Several arguments

`forAllZip` generates and shrinks each argument independently (`forAllZipWith` takes a `Config` first):

```zig
test "addition is commutative" {
    try zigcheck.forAllZip(.{
        zigcheck.generators.int(i32),
        zigcheck.generators.int(i32),
    }, struct {
        fn prop(a: i32, b: i32) !void {
            if (a +% b != b +% a) return error.PropertyFalsified;
        }
    }.prop);
}
```

### Preconditions

`assume()` discards cases that do not meet a precondition; the run gives up after `max_discard` (default 500):

```zig
test "division undoes multiplication" {
    try zigcheck.forAllZip(.{
        zigcheck.generators.intRange(i32, -1000, 1000),
        zigcheck.generators.intRange(i32, -1000, 1000),
    }, struct {
        fn prop(a: i32, b: i32) !void {
            try zigcheck.assume(b != 0);
            if (@divTrunc(a * b, b) != a) return error.PropertyFalsified;
        }
    }.prop);
}
```

## Generators for your types

1. **Primitive or standard fields**: `auto(T)` derives a generator (and per-field shrinking) for ints, floats, bools, enums, structs, `?T`, `[]const T` and `union(enum)`.

   ```zig
   const Point = struct { x: i32, y: i32 };
   try zigcheck.forAll(Point, zigcheck.auto(Point), prop);
   ```

2. **Some fields need constraints**: `build(T, .{ ... })` takes one generator per field in declaration order, with per-field shrinking. This is the default choice for constrained structs.

   ```zig
   const Spec = struct { name: []const u8, arity: u8 };
   const specGen = zigcheck.build(Spec, .{
       zigcheck.sliceOfRange(zigcheck.element(u8, "abcdefgh_"), 1, 12), // name
       zigcheck.generators.intRange(u8, 0, 4), // arity
   });
   ```

3. **Fields depend on each other**: write `Gen(T).fromGenFn(fn (rng, alloc, size) T)`, which does not shrink, or generate independent parts and `assume()` the relation. Decomposing into independent generators with `build` keeps shrinking.

4. **The type carries its own generator**: declare `pub const zigcheck_gen: zigcheck.Gen(T)` on it; `auto(T)` uses that instead of deriving. Prefer `build` for test-only types.

A readable pattern for round trips: generate a small *spec* struct and build the real input from it, so counterexamples show spec fields rather than raw strings.

## Reference

### Primitives

| Generator | Produces |
|---|---|
| `generators.int(T)` | Size-scaled integer (small at size 0, full range at max) |
| `generators.intRange(T, min, max)` | Integer in `[min, max]` |
| `generators.float(T)` | Size-scaled finite float |
| `generators.boolean()` / `generators.byte()` | `bool` / `u8` |
| `generators.positive(T)` / `nonNegative(T)` / `nonZero(T)` / `negative(T)` | Integers with that sign constraint |

### Strings and slices

| Generator | Produces |
|---|---|
| `asciiChar()` / `asciiString(max)` / `asciiStringRange(min, max)` | Printable ASCII (32-126) |
| `alphanumeric()` / `alphanumericString(max)` | `[a-zA-Z0-9]` |
| `string(max)` | Raw bytes |
| `unicodeChar()` / `unicodeString(max_cps)` | Code point (no surrogates) / valid UTF-8 |
| `sliceOf(gen, max)` / `sliceOfRange(gen, min, max)` | Slice of anything `gen` makes; element type inferred |
| `slice(T, gen, max)` / `sliceRange(T, gen, min, max)` | Same with an explicit element type |
| `arrayOf(T, gen, N)` | `[N]T` with per-element shrinking |
| `shuffle(T, items)` / `sublistOf(T, items)` | Permutation / subsequence of a fixed list |
| `orderedList(T, gen, max)` | Sorted slice |
| `growingElements(T, items)` | Element, biased toward earlier ones |

`sliceOf(gen, max)` can shrink to empty; use `sliceOfRange(gen, 1, max)` when the property needs an element.

### Combinators

| Combinator | Description |
|---|---|
| `constant(T, v)`, `element(T, choices)` | Always `v`; pick from a list |
| `oneOf(T, gens)`, `frequency(T, weighted)` | Pick a generator; weighted pick from `{weight, gen}` pairs |
| `map(A, B, gen, fn)`, `mapAlloc` | Transform output (no shrinking); `mapAlloc` gives `fn` the arena allocator |
| `shrinkMap(A, B, gen, fwd, bwd)`, `shrinkMapAlloc` | Transform and keep shrinking through an isomorphism |
| `filter(T, gen, pred)` | Retry up to 1000 times, then warn; prefer `assume()` for restrictive predicates |
| `suchThatMap(A, B, gen, fn)` | Filter and transform; `fn` returns `null` to discard |
| `flatMap(A, B, gen, fn)` | Dependent generation (no shrinking) |
| `build(T, gens)` | Struct from per-field generators |
| `zip(gens)`, `zipMap(gens, R, fn)` | Tuple of generators; zip and map with splatted arguments |
| `noShrink(T, gen)` | Disable shrinking |
| `sized(T, factory)`, `resize(T, gen, n)`, `scale(T, gen, pct)`, `mapSize(T, gen, fn)` | Control the size parameter |
| `funGen(A, B, gen_b)` | Random pure functions (QuickCheck `Fun`) |
| `GenType(gen)` | The `T` of a `Gen(T)` |

```zig
// Weighted choice: 90% small, 10% large
const weighted = comptime zigcheck.frequency(u32, &.{
    .{ 9, zigcheck.generators.intRange(u32, 0, 10) },
    .{ 1, zigcheck.generators.intRange(u32, 1000, 10000) },
});

// Dependent generation: the first value picks the generator for the second
const dependent = comptime zigcheck.flatMap(bool, u32, zigcheck.generators.boolean(), struct {
    fn f(small: bool) zigcheck.Gen(u32) {
        return if (small)
            zigcheck.generators.intRange(u32, 0, 10)
        else
            zigcheck.generators.intRange(u32, 1000, 10000);
    }
}.f);
```

### Shrinking

Every generator ships a shrinker; `.verbose_shrink = true` prints each step.

| Type | Strategy |
|---|---|
| Integer | Binary search toward zero (`intRange`: toward `min`); sign flip for negatives |
| Bool, enum | `true` to `false`; variants with a lower declaration index |
| Float | `0.0`, then halve toward zero (handles NaN and Inf) |
| Struct, union | Each field or payload independently; unions also try earlier variants |
| Slice | Remove chunks (halves, quarters, ..., single elements), then shrink elements |
| `element`, `filter` | Toward earlier elements; inner shrinker filtered by the predicate |

### Size

Like QuickCheck, a `size` parameter ramps from 0 to `max_size` (default 100) across the run, so early cases are small and later ones large. `int`, `float` and the slice and string generators scale with it; `intRange`, `boolean`, `element` and enums do not.

### Configuration

```zig
try zigcheck.forAllWith(.{
    .num_tests = 500,       // default 100
    .max_shrinks = 2000,    // default 1000
    .max_discard = 1000,    // default 500
    .seed = 0x2a,           // default null (time-based); failures print theirs
    .verbose = true,        // print every case
    .verbose_shrink = true, // print every shrink step
    .max_size = 200,        // default 100
    .confidence = 0.99,     // coverage checks, default 0.95
}, i32, gen, property);

const cfg = (zigcheck.Config{}).withNumTests(500).withMaxShrinks(2000).withSeed(0x2a);
```

The runner keeps generated values in an internal arena, so slice and string generators need no allocator setup. A property that allocates should use `std.testing.allocator`.

### Running properties

| Function | Description |
|---|---|
| `forAll(T, gen, prop)`, `forAllWith(cfg, ...)` | Check a property |
| `forAllZip(gens, prop)`, `forAllZipWith(cfg, ...)` | N-argument property |
| `check(cfg, T, gen, prop)`, `recheck(T, gen, prop, result)` | Return a `CheckResult`; replay a failure |
| `expectFailure(T, gen, prop)` | Pass only if the property fails |
| `conjoin(cfg, T, gen, props)`, `disjoin(...)` | All hold; at least one holds |
| `within(T, timeout_us, prop)` | Fail if the property is slower than the limit |
| `assume(cond)`, `assertEqual(T, a, b)`, `counterexample(fmt, args)` | Property helpers |
| `sample(T, gen, n, alloc)`, `sampleWith(..., seed, ...)` | Return `n` generated values, for inspecting a generator |

### Coverage and labels

```zig
try zigcheck.forAllLabeled(i32, zigcheck.generators.int(i32), prop, struct {
    fn classify(n: i32) []const []const u8 {
        if (n > 0) return &.{"positive"};
        if (n < 0) return &.{"negative"};
        return &.{"zero"};
    }
}.classify);
// Prints: 50.2% positive, 49.7% negative, 0.1% zero
```

`forAllCover` adds minimum percentages per label, checked with Wilson score confidence intervals: a label fails only when the interval's upper bound is below the requirement. Also: `forAllLabeledWith`, `checkLabeled`, `forAllCollect` (label by value), `forAllTabulate` (named tables), and `forAllCtx` / `forAllCtxWith` with a `PropertyContext` for composable classify, cover and label.

### Stateful testing

Generate random command sequences against a model (QuickCheck's `Monadic`, Erlang's `eqc_statem`):

```zig
const Command = union(enum) { push: i32, pop };
const Model = struct { size: usize = 0 };

const Spec = zigcheck.StateMachine(Command, Model, *MyStack);
try Spec.runWith(.{ .num_tests = 100, .max_commands = 30 }, .{
    .init_model = initModel,
    .init_sut = initStack,
    .gen_command = genCmd,
    .precondition = precond,
    .run_command = runCmd,
    .next_model = nextModel,
    .postcondition = postCond,
    // .cleanup_sut = freeStack,  // for sockets, arenas, file handles
});
```

Failing sequences shrink in two phases: remove chunks of commands, then shrink command payloads (`push(1000)` to `push(0)`).

## zigcheck and fuzzing

`std.testing.fuzz` feeds raw bytes to a function and uses coverage to find crashes. zigcheck checks invariants on typed, structured inputs and shrinks failures. They answer different questions, so use both:

```zig
// zigcheck: does encode then decode preserve the value?
test "roundtrip preserves data" {
    try zigcheck.forAll(User, zigcheck.auto(User), struct {
        fn prop(user: User) !void {
            const decoded = try decode(try encode(user));
            try std.testing.expectEqual(user.id, decoded.id);
        }
    }.prop);
}
```

**Hybrid mode.** Use the fuzzer as the byte source for structured generation. `forAllFuzzOne` builds a value from the bytes (zeros once exhausted), tests the property, and shrinks on failure; `checkFuzzOne` returns a `FuzzCheckResult` without logging.

```zig
test "roundtrip is coverage-guided" {
    try std.testing.fuzz({}, struct {
        fn f(_: void, smith: *std.testing.Smith) !void {
            // The corpus entry as written, or bytes from the fuzzer.
            var buf: [1024]u8 = undefined;
            const bytes = if (smith.in) |in| in else buf[0..smith.slice(&buf)];
            try zigcheck.forAllFuzzOne(User, zigcheck.auto(User), bytes, struct {
                fn prop(u: User) !void {
                    try std.testing.expectEqual(u.id, (try decode(try encode(u))).id);
                }
            }.prop);
        }
    }.f, .{});
}
```

## Development

```bash
zig build test            # or: just test
just test-filter "shrink" # tests matching a name
just fmt-check            # zig fmt --check src/
just fmt                  # zig fmt src/
just release "0.6.1" "notes"   # annotated tag, push, GitHub release
```

Source layout: `zigcheck.zig` (public API), `gen.zig` (`Gen(T)`), `generators.zig` (primitives), `combinators.zig`, `collections.zig` (slices, strings, lists), `auto.zig` (derivation), `shrink.zig`, `runner.zig` (`forAll` and the shrink loop), `fuzz.zig`, `stateful.zig`, `clock.zig` (time-based seed and `within`).

Issue tracking uses [beads](https://github.com/steveyegge/beads).

## License

MIT
