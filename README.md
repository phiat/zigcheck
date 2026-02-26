# zcheck

[![Zig](https://img.shields.io/badge/Zig-0.15.2-f7a41d?logo=zig&logoColor=white)](https://ziglang.org)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](#running-tests)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-orange)](build.zig.zon)

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
  Shrunk (12 steps)
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
| `generators.float(T)` | `Gen(T)` | Float in `[0, 1)` |
| `generators.boolean()` | `Gen(bool)` | `true` or `false` |
| `generators.byte()` | `Gen(u8)` | Single byte (alias for `int(u8)`) |

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
| `map(A, B, gen, fn)` | `Gen(B)` | Transform output type |
| `filter(T, gen, pred)` | `Gen(T)` | Retry until predicate holds |

```zig
// Only test with positive even numbers
const pos_even = comptime zcheck.filter(i32, zcheck.generators.int(i32), struct {
    fn pred(n: i32) bool {
        return n > 0 and @mod(n, 2) == 0;
    }
}.pred);
```

## Shrinking

Every generator comes with a built-in shrinker that converges toward a minimal counterexample:

| Type | Strategy |
|---|---|
| Integer | Binary search toward zero; try sign flip for negatives |
| Bool | `true` shrinks to `false` |
| Float | Yield `0.0`, then halve toward zero |
| Enum | Yield variants with lower ordinal index |
| Struct | Shrink each field independently |
| `element` | Shrink toward earlier elements in the list |
| `filter` | Inner shrinker, filtered by predicate |

The runner uses an arena allocator that resets each shrink iteration, so shrink state is automatically cleaned up with zero manual `deinit` calls.

## Configuration

```zig
try zcheck.forAllWith(.{
    .num_tests = 500,      // default: 100
    .max_shrinks = 2000,   // default: 1000
    .seed = 0x2a,          // default: null (time-based)
    .verbose = true,       // default: false
}, i32, gen, property);
```

Use `.seed` for deterministic, reproducible test runs. Failed tests print their seed so you can replay them.

## API

### `forAll(T, gen, property) !void`

Run a property check with default config. Integrates with `std.testing` — a failing property becomes a test failure.

### `forAllWith(config, T, gen, property) !void`

Run with explicit `Config`.

### `check(config, T, gen, property) CheckResult(T)`

Run a property check and return the result without failing. Useful when you want to inspect the `CheckResult` programmatically.

```zig
const result = zcheck.check(.{ .seed = 42 }, u32, gen, prop);
switch (result) {
    .passed => |p| std.debug.print("passed {d} tests\n", .{p.num_tests}),
    .failed => |f| std.debug.print("shrunk to {d} in {d} steps\n", .{f.shrunk, f.shrink_steps}),
}
```

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
