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
                // all generators' shrinkers in round-robin order. This gives
                // each shrinker a fair share of the shrink budget rather than
                // exhausting one before trying the next.
                const iters = allocator.alloc(ShrinkIter(T), gens.len) catch return ShrinkIter(T).empty();
                inline for (gens, 0..) |g, i| {
                    iters[i] = g.shrink(value, allocator);
                }

                const State = struct {
                    iters_arr: []ShrinkIter(T),
                    pos: usize,
                    exhausted: usize,
                };
                const state = allocator.create(State) catch {
                    allocator.free(iters);
                    return ShrinkIter(T).empty();
                };
                state.* = .{ .iters_arr = iters, .pos = 0, .exhausted = 0 };

                return .{
                    .context = @ptrCast(state),
                    .nextFn = struct {
                        fn next(ctx: *anyopaque) ?T {
                            const s: *State = @ptrCast(@alignCast(ctx));
                            // Round-robin across shrinkers
                            var tried: usize = 0;
                            while (tried < s.iters_arr.len) {
                                const idx = s.pos % s.iters_arr.len;
                                s.pos += 1;
                                if (s.iters_arr[idx].next()) |val| {
                                    return val;
                                }
                                tried += 1;
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
                std.log.warn("zigcheck.map: shrinking disabled for non-invertible transform; use shrinkMap(A, B, gen, fwd, bwd) for shrink support", .{});
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
                std.log.info("zigcheck.filter: predicate rejected {d} consecutive values; predicate may be too restrictive", .{max_retries});
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
                // We don't know which generator produced the value, so try
                // all generators' shrinkers in round-robin order.
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
                            var tried: usize = 0;
                            while (tried < s.iters_arr.len) {
                                const idx = s.pos % s.iters_arr.len;
                                s.pos += 1;
                                if (s.iters_arr[idx].next()) |val| {
                                    return val;
                                }
                                tried += 1;
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
                std.log.warn("zigcheck.flatMap: shrinking disabled for dependent generation; consider restructuring to use a direct generator with shrinkMap", .{});
                return ShrinkIter(B).empty();
            }
        }.shrinkFn,
    };
}

/// Create a generator whose generation depends on the current size parameter.
/// The factory function receives the current size and returns a generator.
/// QuickCheck: `sized`.
pub fn sized(comptime T: type, comptime factory: *const fn (usize) Gen(T)) Gen(T) {
    return .{
        .genFn = struct {
            fn gen(rng: std.Random, allocator: std.mem.Allocator, size: usize) T {
                const inner = factory(size);
                return inner.generate(rng, allocator, size);
            }
        }.gen,
        .shrinkFn = struct {
            fn shrinkFn(value: T, allocator: std.mem.Allocator) ShrinkIter(T) {
                // Can't recover which inner generator was used, so no shrinking.
                _ = value;
                _ = allocator;
                return ShrinkIter(T).empty();
            }
        }.shrinkFn,
    };
}

/// Override the size parameter for a generator. The inner generator always
/// sees `new_size` regardless of what the runner passes.
/// QuickCheck: `resize`.
pub fn resize(comptime T: type, comptime inner: Gen(T), comptime new_size: usize) Gen(T) {
    return .{
        .genFn = struct {
            fn gen(rng: std.Random, allocator: std.mem.Allocator, _: usize) T {
                return inner.generate(rng, allocator, new_size);
            }
        }.gen,
        .shrinkFn = inner.shrinkFn,
    };
}

/// Scale the size parameter by a comptime-known factor (as a percentage).
/// `scale(T, gen, 50)` halves the size; `scale(T, gen, 200)` doubles it.
/// QuickCheck: `scale`.
pub fn scale(comptime T: type, comptime inner: Gen(T), comptime pct: usize) Gen(T) {
    return .{
        .genFn = struct {
            fn gen(rng: std.Random, allocator: std.mem.Allocator, size: usize) T {
                const scaled = size * pct / 100;
                return inner.generate(rng, allocator, scaled);
            }
        }.gen,
        .shrinkFn = inner.shrinkFn,
    };
}

/// Transform the size parameter with a comptime-known function before passing
/// to the inner generator. QuickCheck: `mapSize`.
///
/// ```zig
/// // Double the size for this generator
/// const g = zigcheck.mapSize(u32, zigcheck.generators.int(u32), struct {
///     fn f(size: usize) usize { return size * 2; }
/// }.f);
/// ```
pub fn mapSize(comptime T: type, comptime inner: Gen(T), comptime f: *const fn (usize) usize) Gen(T) {
    return .{
        .genFn = struct {
            fn gen(rng: std.Random, allocator: std.mem.Allocator, size: usize) T {
                return inner.generate(rng, allocator, f(size));
            }
        }.gen,
        .shrinkFn = inner.shrinkFn,
    };
}

/// Filter and transform in one step. Generates values from `inner`, applies
/// `f`, and keeps only `non-null` results. Retries up to 1000 times.
/// QuickCheck: `suchThatMap`.
///
/// ```zig
/// // Generate even numbers by doubling, with shrinking
/// const g = zigcheck.suchThatMap(u32, u32, zigcheck.generators.int(u32), struct {
///     fn f(n: u32) ?u32 {
///         if (n > 1000) return null;  // reject large values
///         return n * 2;               // transform to even
///     }
/// }.f);
/// ```
pub fn suchThatMap(
    comptime A: type,
    comptime B: type,
    comptime inner: Gen(A),
    comptime f: *const fn (A) ?B,
) Gen(B) {
    const max_retries = 1000;
    return .{
        .genFn = struct {
            fn gen(rng: std.Random, allocator: std.mem.Allocator, size: usize) B {
                for (0..max_retries) |_| {
                    const a = inner.generate(rng, allocator, size);
                    if (f(a)) |b| return b;
                }
                std.log.info("zigcheck.suchThatMap: transform returned null {d} consecutive times", .{max_retries});
                // Last resort: generate one more and force through.
                // If it's still null, return zeroed memory — the property
                // test will likely fail and report a useful counterexample.
                const a = inner.generate(rng, allocator, size);
                return f(a) orelse std.mem.zeroes(B);
            }
        }.gen,
        .shrinkFn = struct {
            fn shrinkFn(_: B, _: std.mem.Allocator) ShrinkIter(B) {
                // Can't shrink: f is not invertible (same as map).
                return ShrinkIter(B).empty();
            }
        }.shrinkFn,
    };
}

// -- Function generation (CoArbitrary equivalent) -------------------------

/// Generate random functions from A to B. QuickCheck: `CoArbitrary a => Arbitrary (a -> b)`.
///
/// Returns `Gen(FunWith(A, B, gen_b))` — each generated function is
/// deterministic and pure. Use `fun.call(a)` to invoke.
///
/// Each generated function is deterministic and pure — the same input always
/// produces the same output for a given Fun instance. Different Fun instances
/// (different seeds) produce different functions.
///
/// Shrinks toward seed 0 (which produces a "simpler" function).
pub fn funGen(comptime A: type, comptime B: type, comptime gen_b: Gen(B)) Gen(FunWith(A, B, gen_b)) {
    return .{
        .genFn = struct {
            fn f(rng: std.Random, _: std.mem.Allocator, _: usize) FunWith(A, B, gen_b) {
                return .{ .seed = rng.int(u64) };
            }
        }.f,
        .shrinkFn = struct {
            fn f(value: FunWith(A, B, gen_b), allocator: std.mem.Allocator) ShrinkIter(FunWith(A, B, gen_b)) {
                // Shrink the seed using integer shrinking
                const state = allocator.create(shrink.IntShrinkState(u64)) catch return ShrinkIter(FunWith(A, B, gen_b)).empty();
                state.* = shrink.IntShrinkState(u64).init(value.seed);

                const State = struct {
                    inner: *shrink.IntShrinkState(u64),

                    fn next(self: *@This()) ?FunWith(A, B, gen_b) {
                        if (self.inner.next()) |new_seed| {
                            return .{ .seed = new_seed };
                        }
                        return null;
                    }

                    fn typeErasedNext(ctx: *anyopaque) ?FunWith(A, B, gen_b) {
                        const self: *@This() = @ptrCast(@alignCast(ctx));
                        return self.next();
                    }
                };
                const wrapper = allocator.create(State) catch return ShrinkIter(FunWith(A, B, gen_b)).empty();
                wrapper.* = .{ .inner = state };
                return .{
                    .context = @ptrCast(wrapper),
                    .nextFn = State.typeErasedNext,
                };
            }
        }.f,
    };
}

/// A random pure function from A to B. Each instance holds a seed; calling
/// `fun.call(a)` hashes the input to perturb the seed and generates a
/// deterministic B value. Different seeds produce different functions.
/// Use `funGen(A, B, gen_b)` to generate these.
pub fn FunWith(comptime A: type, comptime B: type, comptime gen_b: Gen(B)) type {
    return struct {
        seed: u64,

        pub fn call(self: @This(), a: A) B {
            var hasher = std.hash.Wyhash.init(self.seed);
            std.hash.autoHash(&hasher, a);
            var prng = std.Random.DefaultPrng.init(hasher.final());
            var buf: [4096]u8 = undefined;
            var fba = std.heap.FixedBufferAllocator.init(&buf);
            return gen_b.generate(prng.random(), fba.allocator(), 100);
        }

        pub fn format(self: @This(), comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
            try writer.print("Fun(seed=0x{x})", .{self.seed});
        }
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

test "resize: overrides size parameter" {
    var prng = std.Random.DefaultPrng.init(42);
    // resize with size=0 on a size-aware int generator should always produce 0
    const g = resize(i32, generators.int(i32), 0);
    for (0..50) |_| {
        const v = g.generate(prng.random(), std.testing.allocator, 100);
        try std.testing.expectEqual(@as(i32, 0), v);
    }
}

test "scale: halves the size parameter" {
    var prng = std.Random.DefaultPrng.init(42);
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    // scale(50) should halve the effective size; at size 100, effective = 50
    // Use a slice generator where size controls length
    const g = comptime scale([]const u8, @import("collections.zig").slice(u8, generators.int(u8), 100), 50);
    var max_len: usize = 0;
    for (0..100) |_| {
        const v = g.generate(prng.random(), arena_state.allocator(), 100);
        if (v.len > max_len) max_len = v.len;
    }
    // With scale(50), effective size is 50, so max len should be ~50, not 100
    try std.testing.expect(max_len <= 55);
}

test "mapSize: transforms size parameter" {
    var prng = std.Random.DefaultPrng.init(42);
    // mapSize that always returns 0 should make size-aware int produce 0
    const g = comptime mapSize(i32, generators.int(i32), struct {
        fn f(_: usize) usize {
            return 0;
        }
    }.f);
    for (0..50) |_| {
        const v = g.generate(prng.random(), std.testing.allocator, 100);
        try std.testing.expectEqual(@as(i32, 0), v);
    }
}

test "suchThatMap: filters and transforms" {
    var prng = std.Random.DefaultPrng.init(42);
    // Only keep values <= 50, double them
    const g = comptime suchThatMap(u8, u16, generators.int(u8), struct {
        fn f(n: u8) ?u16 {
            if (n > 50) return null;
            return @as(u16, n) * 2;
        }
    }.f);
    for (0..100) |_| {
        const v = g.generate(prng.random(), std.testing.allocator, 100);
        try std.testing.expect(v <= 100);
        try std.testing.expect(v % 2 == 0);
    }
}

test "funGen: generated functions are deterministic" {
    var prng = std.Random.DefaultPrng.init(42);
    const g = funGen(u32, u32, generators.int(u32));
    const f = g.generate(prng.random(), std.testing.allocator, 100);
    // Same input should always produce same output
    const a = f.call(42);
    const b = f.call(42);
    try std.testing.expectEqual(a, b);
}

test "funGen: different inputs produce different outputs (usually)" {
    var prng = std.Random.DefaultPrng.init(42);
    const g = funGen(u32, u32, generators.int(u32));
    const f = g.generate(prng.random(), std.testing.allocator, 100);
    // Different inputs should usually produce different outputs
    var distinct: usize = 0;
    for (0..20) |i| {
        if (f.call(@intCast(i)) != f.call(@intCast(i + 100))) distinct += 1;
    }
    try std.testing.expect(distinct > 10); // most pairs should differ
}

test "funGen: different seeds produce different functions" {
    var prng = std.Random.DefaultPrng.init(42);
    const g = funGen(u32, u32, generators.int(u32));
    const f1 = g.generate(prng.random(), std.testing.allocator, 100);
    const f2 = g.generate(prng.random(), std.testing.allocator, 100);
    // Different functions should disagree on at least some inputs
    var disagree: usize = 0;
    for (0..20) |i| {
        if (f1.call(@intCast(i)) != f2.call(@intCast(i))) disagree += 1;
    }
    try std.testing.expect(disagree > 5);
}
