// Combinators for composing generators.

const std = @import("std");
const Gen = @import("gen.zig").Gen;
const ShrinkIter = @import("shrink.zig").ShrinkIter;
const shrink = @import("shrink.zig");
const generators = @import("generators.zig");

/// Always produces the same value. Shrinks to nothing.
pub fn constant(comptime T: type, comptime value: T) Gen(T) {
    return .{
        .genFn = struct {
            fn f(_: std.Random, _: std.mem.Allocator, _: usize) T {
                return value;
            }
        }.f,
        .shrinkFn = struct {
            fn f(_: T, _: std.mem.Allocator) ShrinkIter(T) {
                return ShrinkIter(T).empty();
            }
        }.f,
    };
}

/// Pick uniformly from a comptime-known list of values.
/// Shrinks toward earlier elements in the list.
pub fn element(comptime T: type, comptime choices: []const T) Gen(T) {
    comptime {
        if (choices.len == 0) @compileError("element() requires at least one choice");
    }
    return .{
        .genFn = struct {
            fn f(rng: std.Random, _: std.mem.Allocator, _: usize) T {
                const idx = rng.intRangeAtMost(usize, 0, choices.len - 1);
                return choices[idx];
            }
        }.f,
        .shrinkFn = struct {
            fn f(value: T, allocator: std.mem.Allocator) ShrinkIter(T) {
                // Find position of current value; shrink to earlier elements
                var current_pos: usize = choices.len;
                for (choices, 0..) |c, i| {
                    if (std.meta.eql(c, value)) {
                        current_pos = i;
                        break;
                    }
                }
                if (current_pos == 0 or current_pos == choices.len) return ShrinkIter(T).empty();

                const candidates = allocator.alloc(T, current_pos) catch return ShrinkIter(T).empty();
                for (choices[0..current_pos], 0..) |c, i| {
                    candidates[i] = c;
                }

                const State = struct {
                    items: []const T,
                    pos: usize,
                };
                const state = allocator.create(State) catch {
                    allocator.free(candidates);
                    return ShrinkIter(T).empty();
                };
                state.* = .{ .items = candidates, .pos = 0 };

                return .{
                    .context = @ptrCast(state),
                    .nextFn = struct {
                        fn next(ctx: *anyopaque) ?T {
                            const s: *State = @ptrCast(@alignCast(ctx));
                            if (s.pos >= s.items.len) return null;
                            const val = s.items[s.pos];
                            s.pos += 1;
                            return val;
                        }
                    }.next,
                };
            }
        }.f,
    };
}

/// Pick from one of several generators uniformly at random.
/// Shrinks using the selected generator's shrinker, then tries earlier generators.
pub fn oneOf(comptime T: type, comptime gens: []const Gen(T)) Gen(T) {
    comptime {
        if (gens.len == 0) @compileError("oneOf() requires at least one generator");
        if (gens.len == 1) return gens[0];
    }
    return .{
        .genFn = struct {
            fn f(rng: std.Random, allocator: std.mem.Allocator, size: usize) T {
                const idx = rng.intRangeAtMost(usize, 0, gens.len - 1);
                inline for (gens, 0..) |g, i| {
                    if (idx == i) return g.generate(rng, allocator, size);
                }
                unreachable;
            }
        }.f,
        .shrinkFn = struct {
            fn f(value: T, allocator: std.mem.Allocator) ShrinkIter(T) {
                // We don't know which generator produced the value, so try
                // all generators' shrinkers. This over-shrinks but is correct.
                const iters = allocator.alloc(ShrinkIter(T), gens.len) catch return ShrinkIter(T).empty();
                inline for (gens, 0..) |g, i| {
                    iters[i] = g.shrink(value, allocator);
                }

                const State = struct {
                    iters_arr: []ShrinkIter(T),
                    pos: usize,
                };
                const state = allocator.create(State) catch {
                    allocator.free(iters);
                    return ShrinkIter(T).empty();
                };
                state.* = .{ .iters_arr = iters, .pos = 0 };

                return .{
                    .context = @ptrCast(state),
                    .nextFn = struct {
                        fn next(ctx: *anyopaque) ?T {
                            const s: *State = @ptrCast(@alignCast(ctx));
                            while (s.pos < s.iters_arr.len) {
                                if (s.iters_arr[s.pos].next()) |val| {
                                    return val;
                                }
                                s.pos += 1;
                            }
                            return null;
                        }
                    }.next,
                };
            }
        }.f,
    };
}

/// Transform generator output. WARNING: shrinking is disabled because `f`
/// is not invertible. Use `shrinkMap` instead if you need shrink support.
pub fn map(
    comptime A: type,
    comptime B: type,
    comptime inner: Gen(A),
    comptime f: *const fn (A) B,
) Gen(B) {
    return .{
        .genFn = struct {
            fn gen(rng: std.Random, allocator: std.mem.Allocator, size: usize) B {
                return f(inner.generate(rng, allocator, size));
            }
        }.gen,
        .shrinkFn = struct {
            fn shrinkFn(_: B, _: std.mem.Allocator) ShrinkIter(B) {
                // f is not invertible, so we can't recover the original A to
                // feed to inner's shrinker. Use shrinkMap for shrink support.
                std.log.warn("zcheck.map: shrinking disabled for non-invertible transform; use shrinkMap(A, B, gen, fwd, bwd) for shrink support", .{});
                return ShrinkIter(B).empty();
            }
        }.shrinkFn,
    };
}

/// Error returned when a filter predicate rejects too many consecutive values.
pub const FilterExhausted = error.FilterExhausted;

/// Filter generated values. Retries up to 1000 times to find a value that
/// satisfies the predicate. Logs a warning if exhausted.
pub fn filter(
    comptime T: type,
    comptime inner: Gen(T),
    comptime pred: *const fn (T) bool,
) Gen(T) {
    const max_retries = 1000;
    return .{
        .genFn = struct {
            fn gen(rng: std.Random, allocator: std.mem.Allocator, size: usize) T {
                for (0..max_retries) |_| {
                    const val = inner.generate(rng, allocator, size);
                    if (pred(val)) return val;
                }
                // Log a diagnostic rather than panicking so the test runner can
                // report the seed and continue with other tests.
                std.log.warn("zcheck.filter: predicate rejected {d} consecutive values; predicate may be too restrictive", .{max_retries});
                // Return the last generated value even though it doesn't pass the
                // predicate. This lets the runner proceed (the property will likely
                // fail, and the seed will be reported).
                return inner.generate(rng, allocator, size);
            }
        }.gen,
        .shrinkFn = struct {
            fn shrinkFn(value: T, allocator: std.mem.Allocator) ShrinkIter(T) {
                // Wrap inner's shrinker but skip candidates that don't pass the predicate
                const inner_iter = allocator.create(ShrinkIter(T)) catch return ShrinkIter(T).empty();
                inner_iter.* = inner.shrink(value, allocator);

                return .{
                    .context = @ptrCast(inner_iter),
                    .nextFn = struct {
                        fn next(ctx: *anyopaque) ?T {
                            const iter: *ShrinkIter(T) = @ptrCast(@alignCast(ctx));
                            while (iter.next()) |candidate| {
                                if (pred(candidate)) return candidate;
                            }
                            return null;
                        }
                    }.next,
                };
            }
        }.shrinkFn,
    };
}

/// Pick from generators with weighted probability.
/// Takes a comptime array of `{weight, gen}` tuples. Higher weights mean
/// more likely to be chosen. Shrinks using the selected generator's shrinker.
pub fn frequency(comptime T: type, comptime weighted: []const struct { usize, Gen(T) }) Gen(T) {
    comptime {
        if (weighted.len == 0) @compileError("frequency() requires at least one weighted generator");
        for (weighted) |entry| {
            if (entry[0] == 0) @compileError("frequency(): weight must be > 0");
        }
    }
    return .{
        .genFn = struct {
            fn gen(rng: std.Random, allocator: std.mem.Allocator, size: usize) T {
                comptime var total: usize = 0;
                inline for (weighted) |entry| {
                    total += entry[0];
                }
                var pick = rng.intRangeLessThan(usize, 0, total);
                inline for (weighted) |entry| {
                    if (pick < entry[0]) return entry[1].generate(rng, allocator, size);
                    pick -= entry[0];
                }
                unreachable;
            }
        }.gen,
        .shrinkFn = struct {
            fn shrinkFn(value: T, allocator: std.mem.Allocator) ShrinkIter(T) {
                // We don't know which generator produced the value, so try all shrinkers
                const iters = allocator.alloc(ShrinkIter(T), weighted.len) catch return ShrinkIter(T).empty();
                inline for (weighted, 0..) |entry, i| {
                    iters[i] = entry[1].shrink(value, allocator);
                }

                const State = struct {
                    iters_arr: []ShrinkIter(T),
                    pos: usize,
                };
                const state = allocator.create(State) catch {
                    allocator.free(iters);
                    return ShrinkIter(T).empty();
                };
                state.* = .{ .iters_arr = iters, .pos = 0 };

                return .{
                    .context = @ptrCast(state),
                    .nextFn = struct {
                        fn next(ctx: *anyopaque) ?T {
                            const s: *State = @ptrCast(@alignCast(ctx));
                            while (s.pos < s.iters_arr.len) {
                                if (s.iters_arr[s.pos].next()) |val| {
                                    return val;
                                }
                                s.pos += 1;
                            }
                            return null;
                        }
                    }.next,
                };
            }
        }.shrinkFn,
    };
}

/// Wrap a generator to disable shrinking. The generated values are
/// produced normally but shrink candidates are never emitted.
pub fn noShrink(comptime T: type, comptime inner: Gen(T)) Gen(T) {
    return .{
        .genFn = inner.genFn,
        .shrinkFn = struct {
            fn f(_: T, _: std.mem.Allocator) ShrinkIter(T) {
                return ShrinkIter(T).empty();
            }
        }.f,
    };
}

/// Shrink via isomorphism: given forward (A -> B) and backward (B -> A) functions,
/// shrink B values by mapping to A, shrinking there, and mapping back.
pub fn shrinkMap(
    comptime A: type,
    comptime B: type,
    comptime inner: Gen(A),
    comptime forward: *const fn (A) B,
    comptime backward: *const fn (B) A,
) Gen(B) {
    return .{
        .genFn = struct {
            fn gen(rng: std.Random, allocator: std.mem.Allocator, size: usize) B {
                return forward(inner.generate(rng, allocator, size));
            }
        }.gen,
        .shrinkFn = struct {
            fn shrinkFn(value: B, allocator: std.mem.Allocator) ShrinkIter(B) {
                // Map B -> A, shrink in A-space, map results back A -> B
                const a_value = backward(value);
                const inner_iter = allocator.create(ShrinkIter(A)) catch return ShrinkIter(B).empty();
                inner_iter.* = inner.shrink(a_value, allocator);

                const State = struct {
                    a_iter: *ShrinkIter(A),
                };
                const state = allocator.create(State) catch {
                    allocator.destroy(inner_iter);
                    return ShrinkIter(B).empty();
                };
                state.* = .{ .a_iter = inner_iter };

                return .{
                    .context = @ptrCast(state),
                    .nextFn = struct {
                        fn next(ctx: *anyopaque) ?B {
                            const s: *State = @ptrCast(@alignCast(ctx));
                            if (s.a_iter.next()) |a_val| {
                                return forward(a_val);
                            }
                            return null;
                        }
                    }.next,
                };
            }
        }.shrinkFn,
    };
}

/// Monadic bind for dependent generation. WARNING: shrinking is disabled
/// because the generator chosen by `f` is not recoverable from a B value.
pub fn flatMap(
    comptime A: type,
    comptime B: type,
    comptime gen_a: Gen(A),
    comptime f: *const fn (A) Gen(B),
) Gen(B) {
    return .{
        .genFn = struct {
            fn gen(rng: std.Random, allocator: std.mem.Allocator, size: usize) B {
                const a = gen_a.generate(rng, allocator, size);
                const gen_b = f(a);
                return gen_b.generate(rng, allocator, size);
            }
        }.gen,
        .shrinkFn = struct {
            fn shrinkFn(_: B, _: std.mem.Allocator) ShrinkIter(B) {
                // Can't shrink: we don't know which gen_b produced this value.
                std.log.warn("zcheck.flatMap: shrinking disabled for dependent generation; consider restructuring to use a direct generator with shrinkMap", .{});
                return ShrinkIter(B).empty();
            }
        }.shrinkFn,
    };
}

// -- Tests ----------------------------------------------------------------

test "constant: always produces the same value" {
    var prng = std.Random.DefaultPrng.init(42);
    const g = constant(u32, 7);
    for (0..50) |_| {
        try std.testing.expectEqual(@as(u32, 7), g.generate(prng.random(), std.testing.allocator, 100));
    }
}

test "element: picks from choices" {
    const choices = [_]u8{ 10, 20, 30 };
    var prng = std.Random.DefaultPrng.init(42);
    const g = element(u8, &choices);
    var seen = [_]bool{ false, false, false };
    for (0..200) |_| {
        const v = g.generate(prng.random(), std.testing.allocator, 100);
        if (v == 10) seen[0] = true;
        if (v == 20) seen[1] = true;
        if (v == 30) seen[2] = true;
    }
    for (seen) |s| try std.testing.expect(s);
}

test "element: shrinks toward earlier elements" {
    const choices = [_]u8{ 10, 20, 30 };
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = element(u8, &choices);
    var si = g.shrink(30, arena_state.allocator());
    try std.testing.expectEqual(@as(u8, 10), si.next().?);
    try std.testing.expectEqual(@as(u8, 20), si.next().?);
    try std.testing.expectEqual(null, si.next());
}

test "element: first element has no shrinks" {
    const choices = [_]u8{ 10, 20, 30 };
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = element(u8, &choices);
    var si = g.shrink(10, arena_state.allocator());
    try std.testing.expectEqual(null, si.next());
}

test "oneOf: picks from multiple generators" {
    var prng = std.Random.DefaultPrng.init(42);
    const g = comptime oneOf(u32, &.{
        constant(u32, 1),
        constant(u32, 2),
        constant(u32, 3),
    });
    var seen = [_]bool{ false, false, false };
    for (0..200) |_| {
        const v = g.generate(prng.random(), std.testing.allocator, 100);
        if (v == 1) seen[0] = true;
        if (v == 2) seen[1] = true;
        if (v == 3) seen[2] = true;
    }
    for (seen) |s| try std.testing.expect(s);
}

test "map: transforms values" {
    const g = map(u32, u64, generators.int(u32), struct {
        fn f(x: u32) u64 {
            return @as(u64, x) * 2;
        }
    }.f);
    var prng = std.Random.DefaultPrng.init(42);
    for (0..50) |_| {
        const v = g.generate(prng.random(), std.testing.allocator, 100);
        try std.testing.expect(v % 2 == 0); // always even
    }
}

test "filter: only produces values satisfying predicate" {
    const g = filter(i32, generators.int(i32), struct {
        fn pred(n: i32) bool {
            return n > 0;
        }
    }.pred);
    var prng = std.Random.DefaultPrng.init(42);
    for (0..100) |_| {
        const v = g.generate(prng.random(), std.testing.allocator, 100);
        try std.testing.expect(v > 0);
    }
}

test "filter: shrinks respect predicate" {
    const g = filter(i32, generators.int(i32), struct {
        fn pred(n: i32) bool {
            return n >= 0;
        }
    }.pred);
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    var si = g.shrink(100, arena_state.allocator());
    // First candidate from int shrinker is 0, which passes pred (>= 0)
    try std.testing.expectEqual(@as(i32, 0), si.next().?);
}

test "filter: shrinks skip values failing predicate" {
    // Filter for even numbers only
    const g = filter(i32, generators.int(i32), struct {
        fn pred(n: i32) bool {
            return @mod(n, 2) == 0;
        }
    }.pred);
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    var si = g.shrink(100, arena_state.allocator());
    // All shrink candidates must be even
    while (si.next()) |candidate| {
        try std.testing.expect(@mod(candidate, 2) == 0);
    }
}

test "frequency: respects weights" {
    var prng = std.Random.DefaultPrng.init(42);
    const g = comptime frequency(u32, &.{
        .{ 9, constant(u32, 1) },
        .{ 1, constant(u32, 2) },
    });
    var count_1: usize = 0;
    var count_2: usize = 0;
    for (0..1000) |_| {
        const v = g.generate(prng.random(), std.testing.allocator, 100);
        if (v == 1) count_1 += 1;
        if (v == 2) count_2 += 1;
    }
    // With 9:1 weights, ~90% should be 1
    try std.testing.expect(count_1 > 700);
    try std.testing.expect(count_2 > 30);
    try std.testing.expect(count_1 + count_2 == 1000);
}

test "frequency: single generator" {
    var prng = std.Random.DefaultPrng.init(42);
    const g = comptime frequency(u32, &.{
        .{ 1, constant(u32, 42) },
    });
    for (0..50) |_| {
        try std.testing.expectEqual(@as(u32, 42), g.generate(prng.random(), std.testing.allocator, 100));
    }
}

test "noShrink: generates normally but has no shrinks" {
    var prng = std.Random.DefaultPrng.init(42);
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = noShrink(i32, generators.int(i32));
    // Generates values
    const v = g.generate(prng.random(), std.testing.allocator, 100);
    _ = v;
    // No shrink candidates
    var si = g.shrink(100, arena_state.allocator());
    try std.testing.expectEqual(null, si.next());
}

test "shrinkMap: shrinks via isomorphism" {
    // Map i32 -> u32 via absolute value. Shrink in i32 space.
    const g = shrinkMap(i32, u32, generators.int(i32), struct {
        fn forward(n: i32) u32 {
            return @intCast(@abs(n));
        }
    }.forward, struct {
        fn backward(n: u32) i32 {
            return @intCast(n);
        }
    }.backward);
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    // Shrink 100 -- should get 0 first (i32 shrinker starts with 0, mapped to u32 0)
    var si = g.shrink(100, arena_state.allocator());
    try std.testing.expectEqual(@as(u32, 0), si.next().?);
}

test "flatMap: dependent generation" {
    // Generate a bool, then generate either 0 or 1000 based on it
    const g = flatMap(bool, u32, generators.boolean(), struct {
        fn f(b: bool) Gen(u32) {
            if (b) return constant(u32, 1000);
            return constant(u32, 0);
        }
    }.f);
    var prng = std.Random.DefaultPrng.init(42);
    var seen_0 = false;
    var seen_1000 = false;
    for (0..200) |_| {
        const v = g.generate(prng.random(), std.testing.allocator, 100);
        if (v == 0) seen_0 = true;
        if (v == 1000) seen_1000 = true;
        try std.testing.expect(v == 0 or v == 1000);
    }
    try std.testing.expect(seen_0);
    try std.testing.expect(seen_1000);
}

test "filter: exhaustion does not panic" {
    // A predicate that always rejects should not panic -- it returns a value
    // and logs a warning instead.
    var prng = std.Random.DefaultPrng.init(42);
    const g = filter(u32, generators.int(u32), struct {
        fn pred(_: u32) bool {
            return false; // always reject
        }
    }.pred);
    // This should NOT panic -- just returns a value
    _ = g.generate(prng.random(), std.testing.allocator, 100);
}
