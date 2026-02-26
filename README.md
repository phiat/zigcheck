# zcheck

[![Zig](https://img.shields.io/badge/Zig-0.15.2-f7a41d?logo=zig&logoColor=white)](https://ziglang.org)
[![Tests](https://img.shields.io/badge/tests-80%2B_passing-brightgreen)](#running-tests)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.0-orange)](build.zig.zon)
[![Generators](https://img.shields.io/badge/generators-25%2B-blueviolet)](#generators)
[![Shrinking](https://img.shields.io/badge/shrinking-automatic-success)](#shrinking)
[![QuickCheck](https://img.shields.io/badge/QuickCheck_parity-~80%25-informational)](#api)

Property-based testing for Zig. Generate random structured inputs, check properties, and automatically shrink failing cases to minimal counterexamples.

## Quick start

```zig
const zcheck = @import("zcheck");

test "addition is commutative" {
    try zcheck.forAll(i32, zcheck.generators.int(i32), struct {
        fn prop(n: i32) !void {
            // This property holds for all integers
            if (n +% 1 != 1 +% n) return error.PropertyFalsified;
        }
    }.prop);
}
```

When a property fails, zcheck automatically shrinks the counterexample to a minimal reproduction:

```
━━━ zcheck: FAILED after 3 tests ━━━━━━━━━━━━━━━━━━━━━━
  Counterexample: 10
  Shrunk (12 steps) from: 1847382901
  Reproduction seed: 0x2a
  Rerun with: .seed = 0x2a
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Installation

Add zcheck as a Zig package dependency in your `build.zig.zon`:

```zig
.dependencies = .{
    .zcheck = .{
        .url = "https://github.com/phiat/zigcheck/archive/main.tar.gz",
        // .hash = "...",  // zig build will tell you the expected hash
    },
},
```

Then in `build.zig`:

```zig
const zcheck_dep = b.dependency("zcheck", .{});
const zcheck_mod = zcheck_dep.module("zcheck");

// Add to your test step:
my_tests.root_module.addImport("zcheck", zcheck_mod);
```

## Generators

### Primitive types

| Generator | Type | Description |
|---|---|---|
| `generators.int(T)` | `Gen(T)` | Full-range integer |
| `generators.intRange(T, min, max)` | `Gen(T)` | Integer in `[min, max]` |
| `generators.float(T)` | `Gen(T)` | Float in `[0, 1)` |
| `generators.boolean()` | `Gen(bool)` | `true` or `false` |
| `generators.byte()` | `Gen(u8)` | Single byte (alias for `int(u8)`) |

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

Slice shrinking tries shorter prefixes first (smallest to largest), then shrinks individual elements. Use `config.allocator` to provide a non-leak-detecting allocator when testing with slice/string generators:

```zig
var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
defer arena.deinit();

try zcheck.forAllWith(.{
    .allocator = arena.allocator(),
}, []const u8, zcheck.asciiString(50), struct {
    fn prop(s: []const u8) !void {
        // test your parser, serializer, etc.
        _ = s;
    }
}.prop);
```

### Derived types

| Generator | Type | Description |
|---|---|---|
| `auto(T)` | `Gen(T)` | Auto-derive from `int`, `float`, `bool`, `enum`, `struct` |

`auto` handles nested structs and enums via comptime reflection:

```zig
const Point = struct { x: i32, y: i32 };
const g = zcheck.auto(Point);
// Generates random Points and shrinks each field independently
```

### Combinators

| Combinator | Signature | Description |
|---|---|---|
| `constant(T, value)` | `Gen(T)` | Always produces `value` |
| `element(T, choices)` | `Gen(T)` | Picks from a fixed list |
| `oneOf(T, gens)` | `Gen(T)` | Picks from multiple generators |
| `frequency(T, weighted)` | `Gen(T)` | Weighted choice from `{weight, gen}` pairs |
| `map(A, B, gen, fn)` | `Gen(B)` | Transform output type |
| `filter(T, gen, pred)` | `Gen(T)` | Retry until predicate holds |
| `flatMap(A, B, gen, fn)` | `Gen(B)` | Monadic bind for dependent generation |
| `noShrink(T, gen)` | `Gen(T)` | Disable shrinking for a generator |
| `shrinkMap(A, B, gen, fwd, bwd)` | `Gen(B)` | Shrink via isomorphism |

```zig
// Only test with positive even numbers
const pos_even = comptime zcheck.filter(i32, zcheck.generators.int(i32), struct {
    fn pred(n: i32) bool {
        return n > 0 and @mod(n, 2) == 0;
    }
}.pred);

// Weighted choice: 90% small, 10% large
const weighted = comptime zcheck.frequency(u32, &.{
    .{ 9, zcheck.generators.intRange(u32, 0, 10) },
    .{ 1, zcheck.generators.intRange(u32, 1000, 10000) },
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
| Bool | `true` shrinks to `false` |
| Float | Yield `0.0`, then halve toward zero |
| Enum | Yield variants with lower declaration index |
| Struct | Shrink each field independently |
| Slice | Shorter prefixes first, then shrink individual elements |
| `element` | Shrink toward earlier elements in the list |
| `filter` | Inner shrinker, filtered by predicate |

The runner uses an arena allocator for shrink state, freed in bulk when shrinking completes. Enable `.verbose_shrink = true` to see each shrink step.

## Multi-argument properties

Test properties involving two or three values with independent shrinking:

```zig
test "addition is commutative" {
    try zcheck.forAll2(i32, i32,
        zcheck.generators.int(i32), zcheck.generators.int(i32),
        struct {
            fn prop(a: i32, b: i32) !void {
                if (a +% b != b +% a) return error.PropertyFalsified;
            }
        }.prop,
    );
}
```

Also available: `forAll2With`, `forAll3`, `forAll3With`, `check2`, `check3`.

## Implication / preconditions

Use `assume()` to discard test cases that don't meet preconditions. The runner tracks discards and gives up if too many are discarded (default: 500):

```zig
test "division is inverse of multiplication" {
    try zcheck.forAll2(i32, i32,
        zcheck.generators.int(i32), zcheck.generators.intRange(i32, 1, 1000),
        struct {
            fn prop(a: i32, b: i32) !void {
                try zcheck.assume(b != 0); // skip division by zero
                const result = @divTrunc(a *% b, b);
                if (result != a) return error.PropertyFalsified;
            }
        }.prop,
    );
}
```

## Coverage / labeling

Track the distribution of generated test cases with `forAllLabeled`:

```zig
try zcheck.forAllLabeled(i32, zcheck.generators.int(i32),
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
try zcheck.forAllWith(.{
    .num_tests = 500,         // default: 100
    .max_shrinks = 2000,      // default: 1000
    .max_discard = 1000,      // default: 500
    .seed = 0x2a,             // default: null (time-based)
    .verbose = true,          // default: false
    .verbose_shrink = true,   // default: false
    .allocator = my_alloc,    // default: std.testing.allocator
}, i32, gen, property);
```

Use `.seed` for deterministic, reproducible test runs. Failed tests print their seed so you can replay them. Use `.allocator` to control memory for generated values -- required for slice/string generators to avoid leak-detection false positives (see [Slices and strings](#slices-and-strings)). Use `.max_discard` to control how many test cases can be discarded via `assume()` before giving up.

## API

### Core

| Function | Description |
|---|---|
| `forAll(T, gen, property)` | Run property check with default config |
| `forAllWith(config, T, gen, property)` | Run with explicit config |
| `forAll2(A, B, gen_a, gen_b, property)` | Two-argument property check |
| `forAll3(A, B, C, gen_a, gen_b, gen_c, property)` | Three-argument property check |
| `check(config, T, gen, property)` | Return `CheckResult` without failing |
| `check2(config, A, B, gen_a, gen_b, property)` | Two-argument check returning result |
| `check3(config, A, B, C, gen_a, gen_b, gen_c, property)` | Three-argument check returning result |

### Property helpers

| Function | Description |
|---|---|
| `assume(condition)` | Discard test case if condition is false |
| `assertEqual(T, expected, actual)` | Assert equality with diagnostic output |
| `counterexample(fmt, args)` | Log context before a property failure |
| `expectFailure(T, gen, property)` | Pass only if the property fails |
| `forAllLabeled(T, gen, property, classifier)` | Collect coverage statistics |

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
  zcheck.zig      # Public API re-exports
  gen.zig          # Gen(T) core type
  generators.zig   # Built-in generators and combinators
  shrink.zig       # ShrinkIter(T) and shrink state types
  runner.zig       # forAll/check engine with shrink loop
```

## License

MIT
