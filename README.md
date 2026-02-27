# zigcheck

[![Zig](https://img.shields.io/badge/Zig-0.15.2-f7a41d?logo=zig&logoColor=white)](https://ziglang.org)
[![Tests](https://img.shields.io/badge/tests-156%2B_passing-brightgreen)](#running-tests)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.2-orange)](build.zig.zon)
[![Generators](https://img.shields.io/badge/generators-40%2B-blueviolet)](#generators)
[![Shrinking](https://img.shields.io/badge/shrinking-automatic-success)](#shrinking)
[![QuickCheck](https://img.shields.io/badge/QuickCheck_parity-~93%25-informational)](#api)

Property-based testing for Zig, inspired by Haskell's [QuickCheck](https://hackage.haskell.org/package/QuickCheck). Generate random structured inputs, check properties, and automatically shrink failing cases to minimal counterexamples.

## Quick start

The #1 property-based testing pattern: **if you encode it, you should be able to decode it back**.

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

This pattern — encode, decode, compare — transfers directly to JSON, protobuf, custom wire formats, anything with a serialization layer.

When a property fails, zigcheck **automatically shrinks** the counterexample to the smallest failing input:

```
--- zigcheck: FAILED after 12 tests ------------------------
  Counterexample: 10
  Shrunk (8 steps) from: 1847382901
  Reproduction seed: 0x2a
  Rerun with: .seed = 0x2a
-------------------------------------------------------------
```

Specialized generators catch bugs in domain-specific code — here, verifying that every Unicode code point roundtrips through UTF-8:

```zig
test "utf-8 encoding roundtrips all code points" {
    try zigcheck.forAll(u21, zigcheck.generators.unicodeChar(), struct {
        fn prop(codepoint: u21) !void {
            var buf: [4]u8 = undefined;
            const len = std.unicode.utf8Encode(codepoint, &buf) catch
                return error.PropertyFalsified;
            const decoded = std.unicode.utf8Decode(buf[0..len]) catch
                return error.PropertyFalsified;
            if (decoded != codepoint) return error.PropertyFalsified;
        }
    }.prop);
}
```

## Installation

Add zigcheck as a Zig package dependency in your `build.zig.zon`:

```zig
.dependencies = .{
    .zigcheck = .{
        .url = "https://github.com/phiat/zigcheck/archive/v0.2.2.tar.gz",
        // .hash = "...",  // zig build will tell you the expected hash
    },
},
```

Then in `build.zig`:

```zig
const zigcheck_dep = b.dependency("zigcheck", .{});
const zigcheck_mod = zigcheck_dep.module("zigcheck");

// Add to your test step:
my_tests.root_module.addImport("zigcheck", zigcheck_mod);
```

## Generators

### Primitive types

| Generator | Type | Description |
|---|---|---|
| `generators.int(T)` | `Gen(T)` | Size-scaled integer (small at size 0, full range at max) |
| `generators.intRange(T, min, max)` | `Gen(T)` | Integer in `[min, max]` |
| `generators.float(T)` | `Gen(T)` | Size-scaled finite float |
| `generators.boolean()` | `Gen(bool)` | `true` or `false` |
| `generators.byte()` | `Gen(u8)` | Single byte (alias for `int(u8)`) |
| `generators.positive(T)` | `Gen(T)` | Strictly positive integer (`> 0`) |
| `generators.nonNegative(T)` | `Gen(T)` | Non-negative integer (`>= 0`) |
| `generators.nonZero(T)` | `Gen(T)` | Non-zero integer (`!= 0`) |
| `generators.negative(T)` | `Gen(T)` | Strictly negative integer (`< 0`, signed only) |

### Slices and strings

| Generator | Type | Description |
|---|---|---|
| `slice(T, gen, max_len)` | `Gen([]const T)` | Slice of `T` with length in `[0, max_len]` |
| `sliceRange(T, gen, min, max)` | `Gen([]const T)` | Slice with length in `[min, max]` |
| `asciiChar()` | `Gen(u8)` | Printable ASCII (32-126) |
| `asciiString(max_len)` | `Gen([]const u8)` | ASCII string up to `max_len` |
| `asciiStringRange(min, max)` | `Gen([]const u8)` | ASCII string with length in `[min, max]` |
| `alphanumeric()` | `Gen(u8)` | `[a-zA-Z0-9]` |
| `alphanumericString(max_len)` | `Gen([]const u8)` | Alphanumeric string |
| `string(max_len)` | `Gen([]const u8)` | Raw bytes (any u8) |
| `unicodeChar()` | `Gen(u21)` | Random Unicode code point (excludes surrogates) |
| `unicodeString(max_cps)` | `Gen([]const u8)` | Valid UTF-8 string up to `max_cps` code points |

Slice shrinking removes chunks (halves, quarters, eighths, ..., single elements), then shrinks individual elements. The runner uses an internal arena for generated values, so no special allocator setup is needed.

**Tip:** Use `sliceOfRange(gen, 1, max)` when your property needs at least one element — `sliceOf(gen, max)` can shrink to empty, which may cause `for (1..s.len)` to overflow.

```zig
try zigcheck.forAll([]const u8, zigcheck.asciiString(50), struct {
    fn prop(s: []const u8) !void {
        // test your parser, serializer, etc.
        _ = s;
    }
}.prop);
```

### Derived types

| Generator | Type | Description |
|---|---|---|
| `auto(T)` | `Gen(T)` | Auto-derive from `int`, `float`, `bool`, `enum`, `struct`, `?T`, `[]const T`, `union(enum)` |

`auto` handles nested structs and enums via comptime reflection:

```zig
const Point = struct { x: i32, y: i32 };
const g = zigcheck.auto(Point);
// Generates random Points and shrinks each field independently
```

Types can override auto-derivation by declaring a `zigcheck_gen` constant:

```zig
const Money = struct {
    cents: u32,
    pub const zigcheck_gen = zigcheck.Gen(Money){
        .genFn = &struct { fn f(rng: std.Random, _: std.mem.Allocator, _: usize) Money {
            return .{ .cents = rng.intRangeAtMost(u32, 0, 100_00) }; // $0-$100
        } }.f,
        .shrinkFn = &struct { fn f(_: Money, _: std.mem.Allocator) @import("zigcheck").ShrinkIter(Money) {
            return @import("zigcheck").ShrinkIter(Money).empty();
        } }.f,
    };
};
// zigcheck.auto(Money) uses zigcheck_gen instead of deriving from fields
```

### Combinators

| Combinator | Signature | Description |
|---|---|---|
| `constant(T, value)` | `Gen(T)` | Always produces `value` |
| `element(T, choices)` | `Gen(T)` | Picks from a fixed list |
| `oneOf(T, gens)` | `Gen(T)` | Picks from multiple generators |
| `frequency(T, weighted)` | `Gen(T)` | Weighted choice from `{weight, gen}` pairs |
| `map(A, B, gen, fn)` | `Gen(B)` | Transform output type (no shrinking; use `shrinkMap`) |
| `filter(T, gen, pred)` | `Gen(T)` | Retry until predicate holds |
| `flatMap(A, B, gen, fn)` | `Gen(B)` | Monadic bind for dependent generation (no shrinking) |
| `noShrink(T, gen)` | `Gen(T)` | Disable shrinking for a generator |
| `shrinkMap(A, B, gen, fwd, bwd)` | `Gen(B)` | Shrink via isomorphism |
| `sized(T, factory)` | `Gen(T)` | Generator from size-dependent factory function |
| `resize(T, gen, size)` | `Gen(T)` | Override size parameter to a fixed value |
| `scale(T, gen, pct)` | `Gen(T)` | Scale size parameter by percentage |
| `mapSize(T, gen, fn)` | `Gen(T)` | Transform size parameter with a function |
| `suchThatMap(A, B, gen, fn)` | `Gen(B)` | Filter and transform in one step |
| `funGen(A, B, gen_b)` | `Gen(FunWith(A,B,gen_b))` | Generate random pure functions (QuickCheck `Fun`) |
| `build(T, gens)` | `Gen(T)` | Struct builder with per-field generators (shrinks independently) |
| `zip(gens)` | `Gen(Tuple)` | Combine generators into a tuple `Gen(struct { A, B, ... })` |
| `arrayOf(T, gen, N)` | `Gen([N]T)` | Fixed-size array with per-element shrinking |
| `zipMap(gens, R, fn)` | `Gen(R)` | Zip generators + map with splatted args |
| `sliceOf(gen, max)` | `Gen([]const T)` | Like `slice` but infers element type |
| `sliceOfRange(gen, min, max)` | `Gen([]const T)` | Like `sliceRange` but infers element type |

```zig
// Only test with positive even numbers
const pos_even = comptime zigcheck.filter(i32, zigcheck.generators.int(i32), struct {
    fn pred(n: i32) bool {
        return n > 0 and @mod(n, 2) == 0;
    }
}.pred);

// Weighted choice: 90% small, 10% large
const weighted = comptime zigcheck.frequency(u32, &.{
    .{ 9, zigcheck.generators.intRange(u32, 0, 10) },
    .{ 1, zigcheck.generators.intRange(u32, 1000, 10000) },
});
```

### Collection generators

| Generator | Signature | Description |
|---|---|---|
| `shuffle(T, items)` | `Gen([]const T)` | Random permutation of a fixed list |
| `sublistOf(T, items)` | `Gen([]const T)` | Random subsequence of a fixed list |
| `orderedList(T, gen, max)` | `Gen([]const T)` | Sorted slice of random values |
| `growingElements(T, items)` | `Gen(T)` | Biased toward earlier elements |

## Shrinking

Every generator comes with a built-in shrinker that converges toward a minimal counterexample:

| Type | Strategy |
|---|---|
| Integer | Binary search toward zero; try sign flip for negatives |
| `intRange` | Binary search toward `min`, clamped to `[min, max]` |
| Bool | `true` shrinks to `false` |
| Float | Yield `0.0`, then halve toward zero (handles NaN/Inf) |
| Enum | Yield variants with lower declaration index |
| Struct | Shrink each field independently |
| Slice | Remove chunks (halves, quarters, ..., single elements), then shrink elements |
| `element` | Shrink toward earlier elements in the list |
| `filter` | Inner shrinker, filtered by predicate |

The runner uses an arena allocator for shrink state, freed in bulk when shrinking completes. Enable `.verbose_shrink = true` to see each shrink step.

## Multi-argument properties

Test properties involving multiple values with independent per-argument shrinking via `forAllZip`:

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

Works with any number of generators. Use `forAllZipWith` for explicit config.

## Implication / preconditions

Use `assume()` to discard test cases that don't meet preconditions. The runner tracks discards and gives up if too many are discarded (default: 500):

```zig
test "division is inverse of multiplication" {
    try zigcheck.forAllZip(.{
        zigcheck.generators.int(i32),
        zigcheck.generators.intRange(i32, 1, 1000),
    }, struct {
        fn prop(a: i32, b: i32) !void {
            try zigcheck.assume(b != 0); // skip division by zero
            const result = @divTrunc(a *% b, b);
            if (result != a) return error.PropertyFalsified;
        }
    }.prop);
}
```

## Coverage / labeling

Track the distribution of generated test cases with `forAllLabeled`:

```zig
try zigcheck.forAllLabeled(i32, zigcheck.generators.int(i32),
    struct {
        fn prop(n: i32) !void {
            if (n == 0) return error.PropertyFalsified;
        }
    }.prop,
    struct {
        fn classify(n: i32) []const []const u8 {
            if (n > 0) return &.{"positive"};
            if (n < 0) return &.{"negative"};
            return &.{"zero"};
        }
    }.classify,
);
// Prints: 50.2% positive, 49.7% negative, 0.1% zero
```

## Configuration

```zig
try zigcheck.forAllWith(.{
    .num_tests = 500,         // default: 100
    .max_shrinks = 2000,      // default: 1000
    .max_discard = 1000,      // default: 500
    .seed = 0x2a,             // default: null (time-based)
    .verbose = true,          // default: false
    .verbose_shrink = true,   // default: false
    .max_size = 200,          // default: 100
    .allocator = my_alloc,    // default: std.testing.allocator
}, i32, gen, property);
```

Use `.seed` for deterministic, reproducible test runs. Failed tests print their seed so you can replay them. The runner uses an internal arena for generated values, so no special allocator setup is needed for slice/string generators. Use `.max_discard` to control how many test cases can be discarded via `assume()` before giving up.

Config supports builder methods for per-property overrides (QuickCheck's `withMaxSuccess`, `withMaxShrinks`, etc.):

```zig
const cfg = (Config{}).withNumTests(500).withMaxShrinks(2000).withSeed(0x2a);
try zigcheck.forAllWith(cfg, i32, gen, property);
```

## Size parameter

Like QuickCheck, zigcheck threads a `size` parameter (0 to `max_size`, default 100) linearly across test cases. Early tests use small values, later tests use large ones. This helps find both small-value edge cases and large-value stress bugs in a single run.

All generators respect size:
- **int(T)** and **float(T)** scale their range — at size 0 they produce `0`, at max size they produce full-range values
- **Slice/string generators** scale their maximum length — at size 0 they generate minimum-length values, at max size they generate up to the configured maximum
- **intRange**, **boolean**, **element**, **enum** ignore size (their range is already constrained)

Use `resize(T, gen, n)` to pin a generator to a fixed size, `scale(T, gen, pct)` to multiply the size by a percentage, or `sized(T, factory)` to build a generator whose behavior depends on the current size. Set `max_size` in Config to change the upper bound.

## API

### Core

| Function | Description |
|---|---|
| `forAll(T, gen, property)` | Run property check with default config |
| `forAllWith(config, T, gen, property)` | Run with explicit config |
| `forAllZip(gens, property)` | N-argument property with splatted args |
| `forAllZipWith(config, gens, property)` | N-argument property with explicit config |
| `check(config, T, gen, property)` | Return `CheckResult` without failing |
| `recheck(T, gen, property, result)` | Replay a failed `CheckResult` (QuickCheck `recheck`) |

### Property helpers

| Function | Description |
|---|---|
| `assume(condition)` | Discard test case if condition is false |
| `assertEqual(T, expected, actual)` | Assert equality with diagnostic output |
| `counterexample(fmt, args)` | Log context before a property failure |
| `expectFailure(T, gen, property)` | Pass only if the property fails |
| `forAllLabeled(T, gen, property, classifier)` | Collect coverage statistics |
| `forAllLabeledWith(config, T, gen, property, classifier)` | Labeled check with explicit config |
| `checkLabeled(config, T, gen, property, classifier, alloc)` | Return `CheckResultLabeled` without failing |
| `forAllCover(config, T, gen, prop, classifier, reqs)` | Labeled check with minimum coverage requirements |
| `forAllCollect(config, T, gen, property)` | Auto-label with stringified value (QuickCheck `collect`) |
| `forAllTabulate(config, T, gen, prop, table, classifier)` | Group labels under a named table (QuickCheck `tabulate`) |
| `conjoin(config, T, gen, properties)` | All properties must hold (`.&&.`) |
| `disjoin(config, T, gen, properties)` | At least one must hold (`.||.`) |
| `within(T, timeout_us, property)` | Fail if property takes longer than limit (QuickCheck `within`) |
| `forAllCtx(T, gen, property)` | Property with `PropertyContext` for composable classify/cover/label |
| `forAllCtxWith(config, T, gen, property)` | Context property with explicit config |

### Stateful testing

Test stateful APIs by generating random command sequences and verifying model invariants (QuickCheck's `Test.QuickCheck.Monadic` / Erlang QuickCheck's `eqc_statem`):

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
});
```

Failing sequences are automatically shrunk by removing chunks of commands (same strategy as slice shrinking), producing minimal counterexamples.

### Utility

| Function | Description |
|---|---|
| `sample(T, gen, n, allocator)` | Generate N sample values for debugging |
| `sampleWith(T, gen, n, seed, allocator)` | Sample with specific seed |

## Running tests

```bash
zig build test
```

## Project structure

```
src/
  zigcheck.zig        # Public API re-exports
  gen.zig            # Gen(T) core type
  generators.zig     # Primitive generators (int, float, bool) + re-exports
  combinators.zig    # Combinators (constant, element, oneOf, map, filter, etc.)
  collections.zig    # Slice, string, unicode, shuffle, sublist, ordered list
  auto.zig           # Auto-derivation (enum, struct, optional, union)
  shrink.zig         # ShrinkIter(T) and shrink state types
  runner.zig         # forAll/check engine with shrink loop
  stateful.zig       # State machine testing (commands, model, postconditions)
```

## Development

Common tasks via [just](https://github.com/casey/just):

```bash
just test           # run all tests
just test-verbose   # run with verbose output
just test-filter "shrink"  # run tests matching a filter
just fmt            # format source files
just stats          # show project stats
just push           # push to all remotes
just clean          # clean build artifacts
```

## Built with

Project management powered by [beads](https://github.com/steveyegge/beads) -- git-backed issue tracking for AI-assisted development.

## License

MIT
