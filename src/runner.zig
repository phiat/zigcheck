// Runner — the forAll/check engine that ties generation, testing, and
// shrinking together.

const std = @import("std");
const Gen = @import("gen.zig").Gen;
const ShrinkIter = @import("shrink.zig").ShrinkIter;
const shrink_mod = @import("shrink.zig");

pub const Config = struct {
    /// Number of test cases to generate.
    num_tests: usize = 100,
    /// Maximum number of shrink attempts.
    max_shrinks: usize = 1000,
    /// RNG seed. null = random (time-based).
    seed: ?u64 = null,
    /// Print each test case as it runs.
    verbose: bool = false,
    /// Allocator for generated values. Defaults to std.testing.allocator.
    allocator: std.mem.Allocator = std.testing.allocator,
};

pub fn CheckResult(comptime T: type) type {
    return union(enum) {
        passed: struct { num_tests: usize },
        failed: struct {
            seed: u64,
            original: T,
            shrunk: T,
            shrink_steps: usize,
            num_tests_before_fail: usize,
        },
    };
}

/// Run a property check with default config. Integrates with std.testing:
/// a failing property becomes a test failure.
pub fn forAll(
    comptime T: type,
    gen: Gen(T),
    property: *const fn (T) anyerror!void,
) !void {
    return forAllWith(.{}, T, gen, property);
}

/// Run a property check with explicit config.
pub fn forAllWith(
    config: Config,
    comptime T: type,
    gen: Gen(T),
    property: *const fn (T) anyerror!void,
) !void {
    const result = check(config, T, gen, property);
    switch (result) {
        .passed => {},
        .failed => |f| {
            std.log.err(
                \\
                \\━━━ zcheck: FAILED after {d} tests ━━━━━━━━━━━━━━━━━━━━━━
                \\
                \\  Counterexample: {any}
                \\  Shrunk ({d} steps) from: {any}
                \\  Reproduction seed: 0x{x}
                \\  Rerun with: .seed = 0x{x}
                \\
                \\━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            , .{ f.num_tests_before_fail, f.shrunk, f.shrink_steps, f.original, f.seed, f.seed });
            return error.PropertyFalsified;
        },
    }
}

/// Run a property check and return the result without failing.
pub fn check(
    config: Config,
    comptime T: type,
    gen: Gen(T),
    property: *const fn (T) anyerror!void,
) CheckResult(T) {
    const seed = config.seed orelse @as(u64, @truncate(@as(u128, @bitCast(std.time.nanoTimestamp()))));
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();

    for (0..config.num_tests) |i| {
        const value = gen.generate(rng, config.allocator);

        if (config.verbose) {
            std.log.info("zcheck: test {d}/{d}", .{ i + 1, config.num_tests });
        }

        // Test the property
        if (property(value)) |_| {
            // passed, continue
        } else |_| {
            // Property failed — attempt shrinking
            const shrunk = doShrink(T, gen, property, value, config.max_shrinks);

            return .{ .failed = .{
                .seed = seed,
                .original = value,
                .shrunk = shrunk.value,
                .shrink_steps = shrunk.steps,
                .num_tests_before_fail = i + 1,
            } };
        }
    }

    return .{ .passed = .{ .num_tests = config.num_tests } };
}

const ShrinkResult = struct {
    fn Of(comptime T: type) type {
        return struct {
            value: T,
            steps: usize,
        };
    }
};

fn doShrink(
    comptime T: type,
    gen: Gen(T),
    property: *const fn (T) anyerror!void,
    original: T,
    max_shrinks: usize,
) ShrinkResult.Of(T) {
    var best = original;
    var steps: usize = 0;

    // Arena is NOT reset between iterations because `best` may hold pointers
    // into arena memory (e.g., if T contains slices). The arena is freed in
    // bulk at the end via defer deinit. Memory growth is bounded by max_shrinks
    // * sizeof(shrink state) which is negligible.
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();

    for (0..max_shrinks) |_| {
        var iter = gen.shrink(best, arena_state.allocator());
        var improved = false;

        while (iter.next()) |candidate| {
            // Does the property still fail with this simpler value?
            if (property(candidate)) |_| {
                // Property passed — this candidate is too simple
            } else |_| {
                // Still fails — this is a better (simpler) counterexample
                best = candidate;
                steps += 1;
                improved = true;
                break; // restart shrinking from the new best
            }
        }

        if (!improved) break; // no simpler counterexample found
    }

    return .{ .value = best, .steps = steps };
}

// ── Tests ───────────────────────────────────────────────────────────────

const generators = @import("generators.zig");

test "forAll: passing property" {
    // A trivially true property
    try forAll(i32, generators.int(i32), struct {
        fn prop(_: i32) !void {}
    }.prop);
}

test "check: passing property returns passed" {
    const result = check(.{ .seed = 42 }, i32, generators.int(i32), struct {
        fn prop(_: i32) !void {}
    }.prop);
    switch (result) {
        .passed => |p| try std.testing.expectEqual(@as(usize, 100), p.num_tests),
        .failed => return error.TestUnexpectedResult,
    }
}

test "check: failing property returns failed" {
    const result = check(.{ .seed = 42, .num_tests = 10 }, u32, generators.int(u32), struct {
        fn prop(n: u32) !void {
            // This will fail for most values
            if (n > 5) return error.PropertyFalsified;
        }
    }.prop);
    switch (result) {
        .passed => return error.TestUnexpectedResult,
        .failed => |f| {
            try std.testing.expectEqual(@as(u64, 42), f.seed);
            try std.testing.expect(f.num_tests_before_fail <= 10);
        },
    }
}

test "check: shrinks n >= 10 to exactly 10" {
    const result = check(.{ .seed = 42, .num_tests = 100 }, u32, generators.int(u32), struct {
        fn prop(n: u32) !void {
            if (n >= 10) return error.PropertyFalsified;
        }
    }.prop);
    switch (result) {
        .passed => return error.TestUnexpectedResult,
        .failed => |f| {
            try std.testing.expectEqual(@as(u32, 10), f.shrunk);
            try std.testing.expect(f.shrink_steps > 0);
        },
    }
}

test "check: shrinks n > 5 to 6" {
    const result = check(.{ .seed = 42, .num_tests = 100 }, u32, generators.int(u32), struct {
        fn prop(n: u32) !void {
            if (n > 5) return error.PropertyFalsified;
        }
    }.prop);
    switch (result) {
        .passed => return error.TestUnexpectedResult,
        .failed => |f| {
            try std.testing.expectEqual(@as(u32, 6), f.shrunk);
        },
    }
}

test "check: deterministic with same seed" {
    const prop = struct {
        fn f(n: i32) !void {
            if (n == 0) return error.PropertyFalsified;
        }
    }.f;

    const r1 = check(.{ .seed = 12345 }, i32, generators.int(i32), prop);
    const r2 = check(.{ .seed = 12345 }, i32, generators.int(i32), prop);

    // Same seed should produce same results
    switch (r1) {
        .passed => switch (r2) {
            .passed => {},
            .failed => return error.TestUnexpectedResult,
        },
        .failed => |f1| switch (r2) {
            .passed => return error.TestUnexpectedResult,
            .failed => |f2| {
                try std.testing.expectEqual(f1.num_tests_before_fail, f2.num_tests_before_fail);
                try std.testing.expectEqual(f1.original, f2.original);
            },
        },
    }
}
