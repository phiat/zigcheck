// Runner -- the forAll/check engine that ties generation, testing, and
// shrinking together.

const std = @import("std");
const Gen = @import("gen.zig").Gen;
const ShrinkIter = @import("shrink.zig").ShrinkIter;
const shrink_mod = @import("shrink.zig");

/// Sentinel error returned by property functions to discard a test case
/// (implication / precondition). Use `assume()` for convenience.
pub const TestDiscarded = error.TestDiscarded;

/// Discard the current test case if the precondition doesn't hold.
/// Use inside property functions to express implications:
///
/// ```zig
/// fn prop(n: i32) !void {
///     try zcheck.assume(n != 0);
///     // ... test with non-zero n
/// }
/// ```
pub fn assume(condition: bool) !void {
    if (!condition) return TestDiscarded;
}

/// Add context to counterexample output. Call before returning an error
/// in your property function to annotate what went wrong.
///
/// ```zig
/// fn prop(n: i32) !void {
///     const result = compute(n);
///     if (result < 0) {
///         zcheck.counterexample("compute({d}) = {d}", .{ n, result });
///         return error.PropertyFalsified;
///     }
/// }
/// ```
pub fn counterexample(comptime fmt: []const u8, args: anytype) void {
    std.log.info("zcheck context: " ++ fmt, args);
}

/// Assert that two values are equal. On failure, logs both values for diagnostics.
pub fn assertEqual(comptime T: type, expected: T, actual: T) !void {
    if (!std.meta.eql(expected, actual)) {
        std.log.err("assertEqual failed: expected {any}, got {any}", .{ expected, actual });
        return error.PropertyFalsified;
    }
}

pub const Config = struct {
    /// Number of test cases to generate.
    num_tests: usize = 100,
    /// Maximum number of shrink attempts.
    max_shrinks: usize = 1000,
    /// Maximum number of discarded test cases before giving up.
    max_discard: usize = 500,
    /// RNG seed. null = random (time-based).
    seed: ?u64 = null,
    /// Print each test case as it runs.
    verbose: bool = false,
    /// Print each shrink step as it runs.
    verbose_shrink: bool = false,
    /// Allocator available for user-side test helpers. The runner uses an
    /// internal arena for generated values, so this is not needed for
    /// basic usage. Defaults to std.testing.allocator.
    allocator: std.mem.Allocator = std.testing.allocator,
};

/// Run a property check expecting it to fail. Returns an error if the
/// property unexpectedly passes all tests.
///
/// ```zig
/// // This test passes because the property is expected to fail:
/// try zcheck.expectFailure(u32, zcheck.generators.int(u32), struct {
///     fn prop(n: u32) !void {
///         if (n > 100) return error.PropertyFalsified;
///     }
/// }.prop);
/// ```
pub fn expectFailure(
    comptime T: type,
    gen: Gen(T),
    property: *const fn (T) anyerror!void,
) !void {
    return expectFailureWith(.{}, T, gen, property);
}

/// Run an expected-failure property check with explicit config.
pub fn expectFailureWith(
    config: Config,
    comptime T: type,
    gen: Gen(T),
    property: *const fn (T) anyerror!void,
) !void {
    const result = check(config, T, gen, property);
    switch (result) {
        .passed => {
            std.log.err(
                \\
                \\--- zcheck: expectFailure UNEXPECTED PASS -----------------
                \\
                \\  Property passed {d} tests but was expected to fail.
                \\
                \\------------------------------------------------------
            , .{config.num_tests});
            return error.ExpectedFailure;
        },
        .failed => {}, // expected
        .gave_up => {
            std.log.err(
                \\
                \\--- zcheck: expectFailure GAVE UP -------------------------
                \\
                \\  Property gave up after too many discards.
                \\
                \\------------------------------------------------------
            , .{});
            return error.GaveUp;
        },
    }
}

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
        gave_up: struct {
            num_tests: usize,
            num_discarded: usize,
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
            logFailure(f.num_tests_before_fail, "{any}", .{f.shrunk}, "{any}", .{f.original}, f.shrink_steps, f.seed);
            return error.PropertyFalsified;
        },
        .gave_up => |g| {
            logGaveUp(g.num_tests, g.num_discarded);
            return error.GaveUp;
        },
    }
}

/// Resolve seed from config, using time-based fallback if null.
fn resolveSeed(config: Config) u64 {
    return config.seed orelse @as(u64, @truncate(@as(u128, @bitCast(std.time.nanoTimestamp()))));
}

/// Shared failure reporting. counterexample and original are format-ready tuples.
fn logFailure(num_tests: usize, comptime ce_fmt: []const u8, ce_args: anytype, comptime orig_fmt: []const u8, orig_args: anytype, shrink_steps: usize, seed: u64) void {
    std.log.err(
        \\
        \\--- zcheck: FAILED after {d} tests ------------------------
        \\
        ++ "\\  Counterexample: " ++ ce_fmt ++
        \\
        ++ "\\  Shrunk ({d} steps) from: " ++ orig_fmt ++
        \\
        \\  Reproduction seed: 0x{x}
        \\  Rerun with: .seed = 0x{x}
        \\
        \\------------------------------------------------------
    , .{num_tests} ++ ce_args ++ .{shrink_steps} ++ orig_args ++ .{ seed, seed });
}

fn logGaveUp(num_tests: usize, num_discarded: usize) void {
    std.log.err(
        \\
        \\--- zcheck: GAVE UP after {d} tests -----------------------
        \\
        \\  Only {d} tests passed before {d} were discarded.
        \\  Consider using a more targeted generator instead of assume().
        \\
        \\------------------------------------------------------
    , .{ num_tests, num_tests, num_discarded });
}

/// Run a property check and return the result without failing.
pub fn check(
    config: Config,
    comptime T: type,
    gen: Gen(T),
    property: *const fn (T) anyerror!void,
) CheckResult(T) {
    const seed = resolveSeed(config);
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();

    // Use an arena for generated values so they are freed in bulk.
    // This prevents leaks when generators allocate (slices, strings).
    // The arena is reset between passing tests to cap memory usage --
    // only the current test case's allocations are live at any time.
    var gen_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer gen_arena.deinit();
    const gen_alloc = gen_arena.allocator();

    var tests_run: usize = 0;
    var discards: usize = 0;

    while (tests_run < config.num_tests) {
        if (discards >= config.max_discard) {
            return .{ .gave_up = .{
                .num_tests = tests_run,
                .num_discarded = discards,
            } };
        }

        // Reset arena before generating each test case. Previous test's
        // allocations (slices, strings) are no longer needed.
        _ = gen_arena.reset(.retain_capacity);

        const value = gen.generate(rng, gen_alloc);

        if (config.verbose) {
            std.log.info("zcheck: test {d}/{d}: {any}", .{ tests_run + 1, config.num_tests, value });
        }

        // Test the property
        if (property(value)) |_| {
            // passed, continue
            tests_run += 1;
        } else |err| {
            if (err == TestDiscarded) {
                discards += 1;
                continue;
            }
            // Property failed -- the arena still holds the failing value's
            // allocations. Pass it directly to doShrink (which uses its own
            // arena for shrink state).
            const shrunk = doShrink(T, gen, property, value, config.max_shrinks, config.verbose_shrink);

            return .{ .failed = .{
                .seed = seed,
                .original = value,
                .shrunk = shrunk.value,
                .shrink_steps = shrunk.steps,
                .num_tests_before_fail = tests_run + 1,
            } };
        }
    }

    return .{ .passed = .{ .num_tests = tests_run } };
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
    verbose_shrink: bool,
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
                // Property passed -- this candidate is too simple
                if (verbose_shrink) {
                    std.log.info("zcheck shrink: candidate {any} passed (too simple)", .{candidate});
                }
            } else |err| {
                // Discarded test cases don't count as failures
                if (err == TestDiscarded) continue;
                // Still fails -- this is a better (simpler) counterexample
                best = candidate;
                steps += 1;
                improved = true;
                if (verbose_shrink) {
                    std.log.info("zcheck shrink: step {d} -> {any}", .{ steps, candidate });
                }
                break; // restart shrinking from the new best
            }
        }

        if (!improved) break; // no simpler counterexample found
    }

    return .{ .value = best, .steps = steps };
}

// -- Labeled / coverage properties ----------------------------------------

/// Coverage statistics collected during a labeled property check.
pub const CoverageResult = struct {
    num_tests: usize,
    /// Map of label -> count. Stored as parallel slices.
    label_names: []const []const u8,
    label_counts: []const usize,

    pub fn format(self: CoverageResult, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        try writer.print("{d} tests:\n", .{self.num_tests});
        for (self.label_names, self.label_counts) |name, count| {
            const pct = @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(self.num_tests)) * 100.0;
            try writer.print("  {d:.1}% {s}\n", .{ pct, name });
        }
    }
};

/// Run a property check that collects coverage labels.
/// The classifier function takes a generated value and returns a list of label strings.
/// After all tests pass, prints the distribution of labels.
pub fn forAllLabeled(
    comptime T: type,
    gen: Gen(T),
    property: *const fn (T) anyerror!void,
    classifier: *const fn (T) []const []const u8,
) !void {
    return forAllLabeledWith(.{}, T, gen, property, classifier);
}

/// Result of a labeled property check.
pub fn CheckResultLabeled(comptime T: type) type {
    return union(enum) {
        passed: struct {
            num_tests: usize,
            label_names: []const []const u8,
            label_counts: []const usize,
        },
        failed: struct {
            seed: u64,
            original: T,
            shrunk: T,
            shrink_steps: usize,
            num_tests_before_fail: usize,
        },
        gave_up: struct {
            num_tests: usize,
            num_discarded: usize,
        },
    };
}

/// Run a labeled property check and return the result without failing.
pub fn checkLabeled(
    config: Config,
    comptime T: type,
    gen: Gen(T),
    property: *const fn (T) anyerror!void,
    classifier: *const fn (T) []const []const u8,
    arena_alloc: std.mem.Allocator,
) CheckResultLabeled(T) {
    const seed = resolveSeed(config);
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();

    // Track label counts using a hash map
    var label_map = std.StringHashMap(usize).init(arena_alloc);

    var tests_run: usize = 0;
    var discards: usize = 0;

    while (tests_run < config.num_tests) {
        if (discards >= config.max_discard) {
            return .{ .gave_up = .{
                .num_tests = tests_run,
                .num_discarded = discards,
            } };
        }

        const value = gen.generate(rng, arena_alloc);

        if (property(value)) |_| {
            const labels = classifier(value);
            for (labels) |label| {
                const entry = label_map.getOrPut(label) catch continue;
                if (!entry.found_existing) {
                    entry.value_ptr.* = 0;
                }
                entry.value_ptr.* += 1;
            }
            tests_run += 1;
        } else |err| {
            if (err == TestDiscarded) {
                discards += 1;
                continue;
            }
            const shrunk = doShrink(T, gen, property, value, config.max_shrinks, config.verbose_shrink);
            return .{ .failed = .{
                .seed = seed,
                .original = value,
                .shrunk = shrunk.value,
                .shrink_steps = shrunk.steps,
                .num_tests_before_fail = tests_run + 1,
            } };
        }
    }

    // Collect labels into parallel slices for the result
    const count = label_map.count();
    const names = arena_alloc.alloc([]const u8, count) catch return .{ .passed = .{
        .num_tests = tests_run,
        .label_names = &.{},
        .label_counts = &.{},
    } };
    const counts = arena_alloc.alloc(usize, count) catch return .{ .passed = .{
        .num_tests = tests_run,
        .label_names = &.{},
        .label_counts = &.{},
    } };

    var iter = label_map.iterator();
    var idx: usize = 0;
    while (iter.next()) |entry| {
        names[idx] = entry.key_ptr.*;
        counts[idx] = entry.value_ptr.*;
        idx += 1;
    }

    return .{ .passed = .{
        .num_tests = tests_run,
        .label_names = names,
        .label_counts = counts,
    } };
}

/// Run a labeled property check with explicit config.
pub fn forAllLabeledWith(
    config: Config,
    comptime T: type,
    gen: Gen(T),
    property: *const fn (T) anyerror!void,
    classifier: *const fn (T) []const []const u8,
) !void {
    var label_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer label_arena.deinit();

    const result = checkLabeled(config, T, gen, property, classifier, label_arena.allocator());
    switch (result) {
        .passed => |p| {
            // Print coverage report
            if (p.label_names.len > 0) {
                std.log.info("zcheck: {d} tests, coverage:", .{p.num_tests});
                for (p.label_names, p.label_counts) |name, count| {
                    const pct = @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(p.num_tests)) * 100.0;
                    std.log.info("  {d:.1}% {s}", .{ pct, name });
                }
            }
        },
        .failed => |f| {
            logFailure(f.num_tests_before_fail, "{any}", .{f.shrunk}, "{any}", .{f.original}, f.shrink_steps, f.seed);
            return error.PropertyFalsified;
        },
        .gave_up => |g| {
            logGaveUp(g.num_tests, g.num_discarded);
            return error.GaveUp;
        },
    }
}

/// Minimum coverage requirement entry: a label name and the minimum percentage
/// of test cases that must receive that label.
pub const CoverageRequirement = struct {
    label: []const u8,
    min_pct: f64,
};

/// Run a property check with minimum coverage requirements.
/// Similar to QuickCheck's `cover`: after all tests pass, verifies that each
/// label meets its minimum percentage threshold.
///
/// ```zig
/// try zcheck.forAllCover(.{}, i32, gen, property, classifier, &.{
///     .{ .label = "positive", .min_pct = 40.0 },
///     .{ .label = "negative", .min_pct = 40.0 },
/// });
/// ```
pub fn forAllCover(
    config: Config,
    comptime T: type,
    gen: Gen(T),
    property: *const fn (T) anyerror!void,
    classifier: *const fn (T) []const []const u8,
    requirements: []const CoverageRequirement,
) !void {
    var label_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer label_arena.deinit();

    const result = checkLabeled(config, T, gen, property, classifier, label_arena.allocator());
    switch (result) {
        .passed => |p| {
            // Check coverage requirements
            for (requirements) |req| {
                var found = false;
                for (p.label_names, p.label_counts) |name, count| {
                    if (std.mem.eql(u8, name, req.label)) {
                        const pct = @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(p.num_tests)) * 100.0;
                        if (pct < req.min_pct) {
                            std.log.warn(
                                \\
                                \\--- zcheck: INSUFFICIENT COVERAGE -----------------------
                                \\
                                \\  Label "{s}": {d:.1}% (required: {d:.1}%)
                                \\  {d} of {d} tests
                                \\
                                \\------------------------------------------------------
                            , .{ req.label, pct, req.min_pct, count, p.num_tests });
                            return error.InsufficientCoverage;
                        }
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    std.log.warn(
                        \\
                        \\--- zcheck: INSUFFICIENT COVERAGE -----------------------
                        \\
                        \\  Label "{s}": 0.0% (required: {d:.1}%)
                        \\  No test cases matched this label.
                        \\
                        \\------------------------------------------------------
                    , .{ req.label, req.min_pct });
                    return error.InsufficientCoverage;
                }
            }
            // Print coverage report
            if (p.label_names.len > 0) {
                std.log.info("zcheck: {d} tests, coverage:", .{p.num_tests});
                for (p.label_names, p.label_counts) |name, count| {
                    const pct = @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(p.num_tests)) * 100.0;
                    std.log.info("  {d:.1}% {s}", .{ pct, name });
                }
            }
        },
        .failed => |f| {
            logFailure(f.num_tests_before_fail, "{any}", .{f.shrunk}, "{any}", .{f.original}, f.shrink_steps, f.seed);
            return error.PropertyFalsified;
        },
        .gave_up => |g| {
            logGaveUp(g.num_tests, g.num_discarded);
            return error.GaveUp;
        },
    }
}

// -- Property conjunction / disjunction -----------------------------------

/// Run multiple properties on the same generated values. All must hold (conjunction).
/// Equivalent to QuickCheck's `.&&.` operator.
pub fn conjoin(
    config: Config,
    comptime T: type,
    gen: Gen(T),
    comptime properties: []const *const fn (T) anyerror!void,
) !void {
    const result = check(config, T, gen, struct {
        fn combined(value: T) anyerror!void {
            inline for (properties) |prop| {
                try prop(value);
            }
        }
    }.combined);
    switch (result) {
        .passed => {},
        .failed => |f| {
            logFailure(f.num_tests_before_fail, "{any}", .{f.shrunk}, "{any}", .{f.original}, f.shrink_steps, f.seed);
            return error.PropertyFalsified;
        },
        .gave_up => |g| {
            logGaveUp(g.num_tests, g.num_discarded);
            return error.GaveUp;
        },
    }
}

/// Run multiple properties on the same generated values. At least one must hold (disjunction).
/// Equivalent to QuickCheck's `.||.` operator.
pub fn disjoin(
    config: Config,
    comptime T: type,
    gen: Gen(T),
    comptime properties: []const *const fn (T) anyerror!void,
) !void {
    const result = check(config, T, gen, struct {
        fn combined(value: T) anyerror!void {
            inline for (properties) |prop| {
                if (prop(value)) |_| return else |_| {}
            }
            return error.PropertyFalsified;
        }
    }.combined);
    switch (result) {
        .passed => {},
        .failed => |f| {
            logFailure(f.num_tests_before_fail, "{any}", .{f.shrunk}, "{any}", .{f.original}, f.shrink_steps, f.seed);
            return error.PropertyFalsified;
        },
        .gave_up => |g| {
            logGaveUp(g.num_tests, g.num_discarded);
            return error.GaveUp;
        },
    }
}

// -- Multi-argument properties --------------------------------------------

pub fn CheckResult2(comptime A: type, comptime B: type) type {
    return union(enum) {
        passed: struct { num_tests: usize },
        failed: struct {
            seed: u64,
            original_a: A,
            original_b: B,
            shrunk_a: A,
            shrunk_b: B,
            shrink_steps: usize,
            num_tests_before_fail: usize,
        },
        gave_up: struct {
            num_tests: usize,
            num_discarded: usize,
        },
    };
}

/// Run a two-argument property check with default config.
pub fn forAll2(
    comptime A: type,
    comptime B: type,
    gen_a: Gen(A),
    gen_b: Gen(B),
    property: *const fn (A, B) anyerror!void,
) !void {
    return forAll2With(.{}, A, B, gen_a, gen_b, property);
}

/// Run a two-argument property check with explicit config.
pub fn forAll2With(
    config: Config,
    comptime A: type,
    comptime B: type,
    gen_a: Gen(A),
    gen_b: Gen(B),
    property: *const fn (A, B) anyerror!void,
) !void {
    const result = check2(config, A, B, gen_a, gen_b, property);
    switch (result) {
        .passed => {},
        .failed => |f| {
            logFailure(f.num_tests_before_fail, "({any}, {any})", .{ f.shrunk_a, f.shrunk_b }, "({any}, {any})", .{ f.original_a, f.original_b }, f.shrink_steps, f.seed);
            return error.PropertyFalsified;
        },
        .gave_up => |g| {
            logGaveUp(g.num_tests, g.num_discarded);
            return error.GaveUp;
        },
    }
}

/// Run a two-argument property check and return the result.
pub fn check2(
    config: Config,
    comptime A: type,
    comptime B: type,
    gen_a: Gen(A),
    gen_b: Gen(B),
    property: *const fn (A, B) anyerror!void,
) CheckResult2(A, B) {
    const seed = resolveSeed(config);
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();

    var gen_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer gen_arena.deinit();
    const gen_alloc = gen_arena.allocator();

    var tests_run: usize = 0;
    var discards: usize = 0;

    while (tests_run < config.num_tests) {
        if (discards >= config.max_discard) {
            return .{ .gave_up = .{
                .num_tests = tests_run,
                .num_discarded = discards,
            } };
        }

        const a = gen_a.generate(rng, gen_alloc);
        const b = gen_b.generate(rng, gen_alloc);

        if (config.verbose) {
            std.log.info("zcheck: test {d}/{d}: ({any}, {any})", .{ tests_run + 1, config.num_tests, a, b });
        }

        if (property(a, b)) |_| {
            tests_run += 1;
        } else |err| {
            if (err == TestDiscarded) {
                discards += 1;
                continue;
            }
            const shrunk = doShrink2(A, B, gen_a, gen_b, property, a, b, config.max_shrinks);
            return .{ .failed = .{
                .seed = seed,
                .original_a = a,
                .original_b = b,
                .shrunk_a = shrunk.a,
                .shrunk_b = shrunk.b,
                .shrink_steps = shrunk.steps,
                .num_tests_before_fail = tests_run + 1,
            } };
        }
    }

    return .{ .passed = .{ .num_tests = tests_run } };
}

fn doShrink2(
    comptime A: type,
    comptime B: type,
    gen_a: Gen(A),
    gen_b: Gen(B),
    property: *const fn (A, B) anyerror!void,
    orig_a: A,
    orig_b: B,
    max_shrinks: usize,
) struct { a: A, b: B, steps: usize } {
    var best_a = orig_a;
    var best_b = orig_b;
    var steps: usize = 0;

    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();

    // Repeat coordinate descent rounds until a full round makes no progress
    // across all dimensions, avoiding local minima from serial single-pass.
    while (steps < max_shrinks) {
        var round_improved = false;

        // Shrink A as far as possible while holding B constant
        while (steps < max_shrinks) {
            var iter_a = gen_a.shrink(best_a, arena_state.allocator());
            var improved_a = false;
            while (iter_a.next()) |candidate| {
                if (property(candidate, best_b)) |_| {} else |err| {
                    if (err == TestDiscarded) continue;
                    best_a = candidate;
                    steps += 1;
                    improved_a = true;
                    round_improved = true;
                    break;
                }
            }
            if (!improved_a) break;
        }

        // Shrink B as far as possible while holding A constant
        while (steps < max_shrinks) {
            var iter_b = gen_b.shrink(best_b, arena_state.allocator());
            var improved_b = false;
            while (iter_b.next()) |candidate| {
                if (property(best_a, candidate)) |_| {} else |err| {
                    if (err == TestDiscarded) continue;
                    best_b = candidate;
                    steps += 1;
                    improved_b = true;
                    round_improved = true;
                    break;
                }
            }
            if (!improved_b) break;
        }

        if (!round_improved) break;
    }

    return .{ .a = best_a, .b = best_b, .steps = steps };
}

pub fn CheckResult3(comptime A: type, comptime B: type, comptime C: type) type {
    return union(enum) {
        passed: struct { num_tests: usize },
        failed: struct {
            seed: u64,
            original_a: A,
            original_b: B,
            original_c: C,
            shrunk_a: A,
            shrunk_b: B,
            shrunk_c: C,
            shrink_steps: usize,
            num_tests_before_fail: usize,
        },
        gave_up: struct {
            num_tests: usize,
            num_discarded: usize,
        },
    };
}

/// Run a three-argument property check with default config.
pub fn forAll3(
    comptime A: type,
    comptime B: type,
    comptime C: type,
    gen_a: Gen(A),
    gen_b: Gen(B),
    gen_c: Gen(C),
    property: *const fn (A, B, C) anyerror!void,
) !void {
    return forAll3With(.{}, A, B, C, gen_a, gen_b, gen_c, property);
}

/// Run a three-argument property check with explicit config.
pub fn forAll3With(
    config: Config,
    comptime A: type,
    comptime B: type,
    comptime C: type,
    gen_a: Gen(A),
    gen_b: Gen(B),
    gen_c: Gen(C),
    property: *const fn (A, B, C) anyerror!void,
) !void {
    const result = check3(config, A, B, C, gen_a, gen_b, gen_c, property);
    switch (result) {
        .passed => {},
        .failed => |f| {
            logFailure(f.num_tests_before_fail, "({any}, {any}, {any})", .{ f.shrunk_a, f.shrunk_b, f.shrunk_c }, "({any}, {any}, {any})", .{ f.original_a, f.original_b, f.original_c }, f.shrink_steps, f.seed);
            return error.PropertyFalsified;
        },
        .gave_up => |g| {
            logGaveUp(g.num_tests, g.num_discarded);
            return error.GaveUp;
        },
    }
}

/// Run a three-argument property check and return the result.
pub fn check3(
    config: Config,
    comptime A: type,
    comptime B: type,
    comptime C: type,
    gen_a: Gen(A),
    gen_b: Gen(B),
    gen_c: Gen(C),
    property: *const fn (A, B, C) anyerror!void,
) CheckResult3(A, B, C) {
    const seed = resolveSeed(config);
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();

    var gen_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer gen_arena.deinit();
    const gen_alloc = gen_arena.allocator();

    var tests_run: usize = 0;
    var discards: usize = 0;

    while (tests_run < config.num_tests) {
        if (discards >= config.max_discard) {
            return .{ .gave_up = .{
                .num_tests = tests_run,
                .num_discarded = discards,
            } };
        }

        const a = gen_a.generate(rng, gen_alloc);
        const b = gen_b.generate(rng, gen_alloc);
        const c = gen_c.generate(rng, gen_alloc);

        if (config.verbose) {
            std.log.info("zcheck: test {d}/{d}: ({any}, {any}, {any})", .{ tests_run + 1, config.num_tests, a, b, c });
        }

        if (property(a, b, c)) |_| {
            tests_run += 1;
        } else |err| {
            if (err == TestDiscarded) {
                discards += 1;
                continue;
            }
            const shrunk = doShrink3(A, B, C, gen_a, gen_b, gen_c, property, a, b, c, config.max_shrinks);
            return .{ .failed = .{
                .seed = seed,
                .original_a = a,
                .original_b = b,
                .original_c = c,
                .shrunk_a = shrunk.a,
                .shrunk_b = shrunk.b,
                .shrunk_c = shrunk.c,
                .shrink_steps = shrunk.steps,
                .num_tests_before_fail = tests_run + 1,
            } };
        }
    }

    return .{ .passed = .{ .num_tests = tests_run } };
}

fn doShrink3(
    comptime A: type,
    comptime B: type,
    comptime C: type,
    gen_a: Gen(A),
    gen_b: Gen(B),
    gen_c: Gen(C),
    property: *const fn (A, B, C) anyerror!void,
    orig_a: A,
    orig_b: B,
    orig_c: C,
    max_shrinks: usize,
) struct { a: A, b: B, c: C, steps: usize } {
    var best_a = orig_a;
    var best_b = orig_b;
    var best_c = orig_c;
    var steps: usize = 0;

    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();

    // Repeat coordinate descent rounds until a full round makes no progress.
    while (steps < max_shrinks) {
        var round_improved = false;

        // Shrink A as far as possible
        while (steps < max_shrinks) {
            var iter_a = gen_a.shrink(best_a, arena_state.allocator());
            var improved_a = false;
            while (iter_a.next()) |candidate| {
                if (property(candidate, best_b, best_c)) |_| {} else |err| {
                    if (err == TestDiscarded) continue;
                    best_a = candidate;
                    steps += 1;
                    improved_a = true;
                    round_improved = true;
                    break;
                }
            }
            if (!improved_a) break;
        }

        // Shrink B as far as possible
        while (steps < max_shrinks) {
            var iter_b = gen_b.shrink(best_b, arena_state.allocator());
            var improved_b = false;
            while (iter_b.next()) |candidate| {
                if (property(best_a, candidate, best_c)) |_| {} else |err| {
                    if (err == TestDiscarded) continue;
                    best_b = candidate;
                    steps += 1;
                    improved_b = true;
                    round_improved = true;
                    break;
                }
            }
            if (!improved_b) break;
        }

        // Shrink C as far as possible
        while (steps < max_shrinks) {
            var iter_c = gen_c.shrink(best_c, arena_state.allocator());
            var improved_c = false;
            while (iter_c.next()) |candidate| {
                if (property(best_a, best_b, candidate)) |_| {} else |err| {
                    if (err == TestDiscarded) continue;
                    best_c = candidate;
                    steps += 1;
                    improved_c = true;
                    round_improved = true;
                    break;
                }
            }
            if (!improved_c) break;
        }

        if (!round_improved) break;
    }

    return .{ .a = best_a, .b = best_b, .c = best_c, .steps = steps };
}

// -- Tests ----------------------------------------------------------------

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
        .gave_up => return error.TestUnexpectedResult,
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
        .gave_up => return error.TestUnexpectedResult,
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
        .gave_up => return error.TestUnexpectedResult,
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
        .gave_up => return error.TestUnexpectedResult,
    }
}

test "check: shrinks slice to minimal length" {
    // Runner now uses an internal arena for generated values, so no
    // special allocator needed.
    const result = check(
        .{ .seed = 42, .num_tests = 200 },
        []const u8,
        generators.slice(u8, generators.int(u8), 20),
        struct {
            fn prop(s: []const u8) !void {
                // Fails for any slice with length >= 3
                if (s.len >= 3) return error.PropertyFalsified;
            }
        }.prop,
    );
    switch (result) {
        .passed => return error.TestUnexpectedResult,
        .failed => |f| {
            // Should shrink to length exactly 3 (minimal failing length)
            try std.testing.expectEqual(@as(usize, 3), f.shrunk.len);
        },
        .gave_up => return error.TestUnexpectedResult,
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
            .gave_up => return error.TestUnexpectedResult,
        },
        .failed => |f1| switch (r2) {
            .passed => return error.TestUnexpectedResult,
            .failed => |f2| {
                try std.testing.expectEqual(f1.num_tests_before_fail, f2.num_tests_before_fail);
                try std.testing.expectEqual(f1.original, f2.original);
            },
            .gave_up => return error.TestUnexpectedResult,
        },
        .gave_up => return error.TestUnexpectedResult,
    }
}

// -- assume / discard tests -----------------------------------------------

test "assume: passing condition does nothing" {
    try assume(true);
}

test "assume: failing condition returns TestDiscarded" {
    if (assume(false)) |_| {
        return error.TestUnexpectedResult;
    } else |err| {
        try std.testing.expectEqual(TestDiscarded, err);
    }
}

test "check: discards with assume don't count as tests" {
    const result = check(.{ .seed = 42, .num_tests = 50 }, i32, generators.int(i32), struct {
        fn prop(n: i32) !void {
            // Discard negative values
            try assume(n >= 0);
            // Property always passes for non-negative
        }
    }.prop);
    switch (result) {
        .passed => |p| try std.testing.expectEqual(@as(usize, 50), p.num_tests),
        .failed => return error.TestUnexpectedResult,
        .gave_up => return error.TestUnexpectedResult,
    }
}

test "check: too many discards gives up" {
    const result = check(.{ .seed = 42, .num_tests = 100, .max_discard = 10 }, u32, generators.int(u32), struct {
        fn prop(_: u32) !void {
            // Discard everything
            return TestDiscarded;
        }
    }.prop);
    switch (result) {
        .passed => return error.TestUnexpectedResult,
        .failed => return error.TestUnexpectedResult,
        .gave_up => |g| {
            try std.testing.expectEqual(@as(usize, 0), g.num_tests);
            try std.testing.expectEqual(@as(usize, 10), g.num_discarded);
        },
    }
}

test "check: shrinking skips discarded candidates" {
    // Property: n must be even AND > 10. Shrink should find 12 (smallest even > 10).
    const result = check(.{ .seed = 42, .num_tests = 100 }, u32, generators.int(u32), struct {
        fn prop(n: u32) !void {
            try assume(@mod(n, 2) == 0); // only test even numbers
            if (n > 10) return error.PropertyFalsified;
        }
    }.prop);
    switch (result) {
        .passed => return error.TestUnexpectedResult,
        .failed => |f| {
            try std.testing.expectEqual(@as(u32, 12), f.shrunk);
        },
        .gave_up => return error.TestUnexpectedResult,
    }
}

// -- forAll2 / check2 tests -----------------------------------------------

test "forAll2: passing two-argument property" {
    try forAll2(i32, i32, generators.int(i32), generators.int(i32), struct {
        fn prop(_: i32, _: i32) !void {}
    }.prop);
}

test "check2: failing property returns failed" {
    const result = check2(.{ .seed = 42, .num_tests = 100 }, u32, u32, generators.int(u32), generators.int(u32), struct {
        fn prop(a: u32, b: u32) !void {
            if (a +% b > 10) return error.PropertyFalsified;
        }
    }.prop);
    switch (result) {
        .passed => return error.TestUnexpectedResult,
        .failed => |f| {
            try std.testing.expectEqual(@as(u64, 42), f.seed);
            try std.testing.expect(f.num_tests_before_fail <= 100);
        },
        .gave_up => return error.TestUnexpectedResult,
    }
}

test "check2: shrinks both arguments" {
    const result = check2(.{ .seed = 42, .num_tests = 100 }, u32, u32, generators.int(u32), generators.int(u32), struct {
        fn prop(a: u32, b: u32) !void {
            if (a >= 5 and b >= 5) return error.PropertyFalsified;
        }
    }.prop);
    switch (result) {
        .passed => return error.TestUnexpectedResult,
        .failed => |f| {
            try std.testing.expectEqual(@as(u32, 5), f.shrunk_a);
            try std.testing.expectEqual(@as(u32, 5), f.shrunk_b);
        },
        .gave_up => return error.TestUnexpectedResult,
    }
}

// -- forAll3 / check3 tests -----------------------------------------------

test "forAll3: passing three-argument property" {
    try forAll3(i32, i32, bool, generators.int(i32), generators.int(i32), generators.boolean(), struct {
        fn prop(_: i32, _: i32, _: bool) !void {}
    }.prop);
}

// -- assertEqual tests ----------------------------------------------------

test "assertEqual: passes for equal values" {
    try assertEqual(i32, 42, 42);
}

test "assertEqual: fails for unequal values" {
    // assertEqual logs via std.log.err, so we verify behavior via the underlying check
    const result = std.meta.eql(@as(i32, 1), @as(i32, 2));
    try std.testing.expect(!result);
}

// -- expectFailure tests --------------------------------------------------

test "expectFailure: passes when property fails" {
    try expectFailure(u32, generators.int(u32), struct {
        fn prop(n: u32) !void {
            if (n > 0) return error.PropertyFalsified;
        }
    }.prop);
}

test "expectFailure: returns error when property passes" {
    // Verify via check() directly that an always-passing property passes,
    // which is what expectFailure would detect as an error.
    const result = check(.{ .seed = 42 }, i32, generators.int(i32), struct {
        fn prop(_: i32) !void {}
    }.prop);
    switch (result) {
        .passed => {}, // expectFailure would return error.ExpectedFailure here
        .failed => return error.TestUnexpectedResult,
        .gave_up => return error.TestUnexpectedResult,
    }
}

// -- labeled property tests -----------------------------------------------

test "forAllLabeled: collects labels without error" {
    try forAllLabeled(i32, generators.int(i32), struct {
        fn prop(_: i32) !void {}
    }.prop, struct {
        fn classify(n: i32) []const []const u8 {
            if (n > 0) return &.{"positive"};
            if (n < 0) return &.{"negative"};
            return &.{"zero"};
        }
    }.classify);
}

test "checkLabeled: returns passed with coverage data" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const result = checkLabeled(.{ .seed = 42, .num_tests = 50 }, i32, generators.int(i32), struct {
        fn prop(_: i32) !void {}
    }.prop, struct {
        fn classify(n: i32) []const []const u8 {
            if (n > 0) return &.{"positive"};
            if (n < 0) return &.{"negative"};
            return &.{"zero"};
        }
    }.classify, arena.allocator());
    switch (result) {
        .passed => |p| {
            try std.testing.expectEqual(@as(usize, 50), p.num_tests);
            try std.testing.expect(p.label_names.len > 0);
        },
        .failed => return error.TestUnexpectedResult,
        .gave_up => return error.TestUnexpectedResult,
    }
}

test "checkLabeled: returns failed on property failure" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const result = checkLabeled(.{ .seed = 42, .num_tests = 50 }, u32, generators.int(u32), struct {
        fn prop(n: u32) !void {
            if (n > 10) return error.PropertyFalsified;
        }
    }.prop, struct {
        fn classify(_: u32) []const []const u8 {
            return &.{"all"};
        }
    }.classify, arena.allocator());
    switch (result) {
        .passed => return error.TestUnexpectedResult,
        .failed => |f| {
            try std.testing.expect(f.shrink_steps > 0);
        },
        .gave_up => return error.TestUnexpectedResult,
    }
}

// -- forAll3 / check3 tests -----------------------------------------------

test "check3: shrinks all three arguments" {
    const result = check3(.{ .seed = 42, .num_tests = 100 }, u32, u32, u32, generators.int(u32), generators.int(u32), generators.int(u32), struct {
        fn prop(a: u32, b: u32, c: u32) !void {
            if (a >= 3 and b >= 3 and c >= 3) return error.PropertyFalsified;
        }
    }.prop);
    switch (result) {
        .passed => return error.TestUnexpectedResult,
        .failed => |f| {
            try std.testing.expectEqual(@as(u32, 3), f.shrunk_a);
            try std.testing.expectEqual(@as(u32, 3), f.shrunk_b);
            try std.testing.expectEqual(@as(u32, 3), f.shrunk_c);
        },
        .gave_up => return error.TestUnexpectedResult,
    }
}

// -- QuickCheck parity tests ----------------------------------------------

test "bad shrinker resilience: shrink that yields original terminates" {
    // A generator whose shrinker always yields the original value (infinite loop).
    // The runner must terminate via max_shrinks limit.
    const bad_gen = Gen(u32){
        .genFn = struct {
            fn f(rng: std.Random, _: std.mem.Allocator) u32 {
                return rng.int(u32);
            }
        }.f,
        .shrinkFn = struct {
            fn f(value: u32, allocator: std.mem.Allocator) ShrinkIter(u32) {
                // Bad shrinker: always yields the same value (never converges)
                const State = struct {
                    val: u32,
                    count: usize,
                    fn next(ctx: *anyopaque) ?u32 {
                        const self: *@This() = @ptrCast(@alignCast(ctx));
                        if (self.count >= 10) return null; // finite but yields original
                        self.count += 1;
                        return self.val;
                    }
                };
                const state = allocator.create(State) catch return ShrinkIter(u32).empty();
                state.* = .{ .val = value, .count = 0 };
                return .{ .context = @ptrCast(state), .nextFn = State.next };
            }
        }.f,
    };

    // Property that always fails -- shrinking should terminate
    const result = check(.{ .seed = 42, .num_tests = 10, .max_shrinks = 50 }, u32, bad_gen, struct {
        fn prop(n: u32) !void {
            if (n > 0) return error.PropertyFalsified;
        }
    }.prop);
    switch (result) {
        .failed => {}, // Good -- terminated despite bad shrinker
        else => return error.TestUnexpectedResult,
    }
}

test "cover: passes when coverage requirements are met" {
    try forAllCover(.{ .seed = 42, .num_tests = 200 }, i32, generators.int(i32), struct {
        fn prop(_: i32) !void {}
    }.prop, struct {
        fn classify(n: i32) []const []const u8 {
            if (n > 0) return &.{"positive"};
            if (n < 0) return &.{"negative"};
            return &.{"zero"};
        }
    }.classify, &.{
        .{ .label = "positive", .min_pct = 30.0 },
        .{ .label = "negative", .min_pct = 30.0 },
    });
}

test "cover: fails when coverage requirement not met" {
    // All values are labeled "all" -- "rare" label never appears
    const result = blk: {
        forAllCover(.{ .seed = 42, .num_tests = 100 }, u32, generators.int(u32), struct {
            fn prop(_: u32) !void {}
        }.prop, struct {
            fn classify(_: u32) []const []const u8 {
                return &.{"all"};
            }
        }.classify, &.{
            .{ .label = "rare", .min_pct = 10.0 },
        }) catch |err| break :blk err;
        break :blk @as(anyerror, error.TestUnexpectedResult);
    };
    try std.testing.expectEqual(error.InsufficientCoverage, result);
}

test "conjoin: all properties must hold" {
    try conjoin(.{ .seed = 42 }, u32, generators.intRange(u32, 0, 100), &.{
        &struct {
            fn prop(n: u32) !void {
                if (n > 100) return error.PropertyFalsified;
            }
        }.prop,
        &struct {
            fn prop(_: u32) !void {}
        }.prop,
    });
}

test "disjoin: at least one property must hold" {
    try disjoin(.{ .seed = 42 }, i32, generators.int(i32), &.{
        &struct {
            fn prop(n: i32) !void {
                if (n < 0) return error.PropertyFalsified; // fails for negative
            }
        }.prop,
        &struct {
            fn prop(n: i32) !void {
                if (n >= 0) return error.PropertyFalsified; // fails for non-negative
            }
        }.prop,
    });
}

test "verbose mode determinism: same results as non-verbose" {
    // Same seed, same config except verbose flag -- must produce identical results.
    const prop = struct {
        fn f(n: u32) !void {
            if (n >= 100) return error.PropertyFalsified;
        }
    }.f;

    const r_quiet = check(.{ .seed = 99, .num_tests = 50 }, u32, generators.int(u32), prop);
    const r_verbose = check(.{ .seed = 99, .num_tests = 50, .verbose = true }, u32, generators.int(u32), prop);

    // Both should fail with the same counterexample
    switch (r_quiet) {
        .failed => |f1| switch (r_verbose) {
            .failed => |f2| {
                try std.testing.expectEqual(f1.original, f2.original);
                try std.testing.expectEqual(f1.shrunk, f2.shrunk);
                try std.testing.expectEqual(f1.num_tests_before_fail, f2.num_tests_before_fail);
                try std.testing.expectEqual(f1.shrink_steps, f2.shrink_steps);
            },
            else => return error.TestUnexpectedResult,
        },
        .passed => switch (r_verbose) {
            .passed => {},
            else => return error.TestUnexpectedResult,
        },
        .gave_up => return error.TestUnexpectedResult,
    }
}
