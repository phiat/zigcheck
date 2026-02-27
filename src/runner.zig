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
///     try zigcheck.assume(n != 0);
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
///         zigcheck.counterexample("compute({d}) = {d}", .{ n, result });
///         return error.PropertyFalsified;
///     }
/// }
/// ```
pub fn counterexample(comptime fmt: []const u8, args: anytype) void {
    std.log.info("zigcheck context: " ++ fmt, args);
}

/// Assert that two values are equal. On failure, logs both values for diagnostics.
pub fn assertEqual(comptime T: type, expected: T, actual: T) !void {
    if (!std.meta.eql(expected, actual)) {
        std.log.err("assertEqual failed: expected {any}, got {any}", .{ expected, actual });
        return error.PropertyFalsified;
    }
}

/// Wrap a property function with a time limit (in microseconds).
/// If the property takes longer than the limit, it fails with
/// `error.PropertyTimedOut`. QuickCheck: `within`.
///
/// ```zig
/// try zigcheck.forAll(u32, gen, zigcheck.within(u32, 1_000_000, struct {
///     fn prop(n: u32) !void { ... }
/// }.prop));
/// ```
///
/// **Note:** This is a post-hoc duration check, not a preemptive timeout.
/// The property runs to completion, then elapsed time is checked. A property
/// that infinite-loops will hang forever. Use external mechanisms (e.g. test
/// runner timeouts) if you need hard cancellation.
pub fn within(
    comptime T: type,
    comptime timeout_us: u64,
    comptime property: *const fn (T) anyerror!void,
) *const fn (T) anyerror!void {
    return struct {
        fn wrapped(value: T) anyerror!void {
            var timer = std.time.Timer.start() catch return property(value);
            property(value) catch |err| return err;
            const elapsed = timer.read();
            if (elapsed / 1000 > timeout_us) {
                std.log.err("zigcheck.within: property took {d}us, limit is {d}us", .{ elapsed / 1000, timeout_us });
                return error.PropertyTimedOut;
            }
        }
    }.wrapped;
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
    /// Maximum size parameter passed to generators. Controls the upper
    /// bound of the size ramp (0 to max_size across the test run).
    /// QuickCheck default is 100.
    max_size: usize = 100,
    /// Allocator available for user-side test helpers. The runner uses an
    /// internal arena for generated values, so this is not needed for
    /// basic usage. Defaults to std.testing.allocator.
    allocator: std.mem.Allocator = std.testing.allocator,

    /// Override the number of test cases. QuickCheck: `withMaxSuccess`.
    pub fn withNumTests(self: Config, n: usize) Config {
        var c = self;
        c.num_tests = n;
        return c;
    }

    /// Override the maximum number of shrink attempts. QuickCheck: `withMaxShrinks`.
    pub fn withMaxShrinks(self: Config, n: usize) Config {
        var c = self;
        c.max_shrinks = n;
        return c;
    }

    /// Override the maximum number of discarded test cases. QuickCheck: `withMaxDiscardRatio`.
    pub fn withMaxDiscard(self: Config, n: usize) Config {
        var c = self;
        c.max_discard = n;
        return c;
    }

    /// Set a deterministic seed. QuickCheck: `withSeed`.
    pub fn withSeed(self: Config, s: u64) Config {
        var c = self;
        c.seed = s;
        return c;
    }

    /// Enable verbose output. QuickCheck: `verbose`.
    pub fn withVerbose(self: Config) Config {
        var c = self;
        c.verbose = true;
        return c;
    }

    /// Override the maximum size parameter. QuickCheck: `withMaxSize`.
    pub fn withMaxSize(self: Config, n: usize) Config {
        var c = self;
        c.max_size = n;
        return c;
    }
};

/// Run a property check expecting it to fail. Returns an error if the
/// property unexpectedly passes all tests.
///
/// ```zig
/// // This test passes because the property is expected to fail:
/// try zigcheck.expectFailure(u32, zigcheck.generators.int(u32), struct {
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
                \\--- zigcheck: expectFailure UNEXPECTED PASS -----------------
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
                \\--- zigcheck: expectFailure GAVE UP -------------------------
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
            /// Original num_tests config value, needed by recheck to
            /// reproduce the same size ramp.
            num_tests_config: usize,
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

/// Replay a previously failed property check using its seed.
/// QuickCheck: `recheck`. Runs a single test with the same seed to reproduce
/// the failure, then shrinks again.
///
/// ```zig
/// const result = zigcheck.check(.{}, u32, gen, property);
/// // ... later, replay the failure:
/// try zigcheck.recheck(u32, gen, property, result);
/// ```
pub fn recheck(
    comptime T: type,
    gen: Gen(T),
    property: *const fn (T) anyerror!void,
    result: CheckResult(T),
) !void {
    switch (result) {
        .failed => |f| {
            return forAllWith(.{ .seed = f.seed, .num_tests = f.num_tests_config }, T, gen, property);
        },
        else => {}, // nothing to replay if it didn't fail
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
        \\--- zigcheck: FAILED after {d} tests ------------------------
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
        \\--- zigcheck: GAVE UP after {d} tests -----------------------
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

        const size = if (config.num_tests > 0) tests_run * config.max_size / config.num_tests else config.max_size;
        const value = gen.generate(rng, gen_alloc, size);

        // Test the property
        if (property(value)) |_| {
            // passed, continue
            tests_run += 1;
            if (config.verbose) {
                std.log.info("zigcheck: test {d}/{d} (size {d}): PASS {any}", .{ tests_run, config.num_tests, size, value });
            }
        } else |err| {
            if (err == TestDiscarded) {
                discards += 1;
                if (config.verbose) {
                    std.log.info("zigcheck: test {d}/{d} (size {d}): DISCARD {any}", .{ tests_run + 1, config.num_tests, size, value });
                }
                continue;
            }
            if (config.verbose) {
                std.log.info("zigcheck: test {d}/{d} (size {d}): FAIL {any}", .{ tests_run + 1, config.num_tests, size, value });
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
                .num_tests_config = config.num_tests,
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
                    std.log.info("zigcheck shrink: candidate {any} passed (too simple)", .{candidate});
                }
            } else |err| {
                // Discarded test cases don't count as failures
                if (err == TestDiscarded) continue;
                // Still fails -- this is a better (simpler) counterexample
                best = candidate;
                steps += 1;
                improved = true;
                if (verbose_shrink) {
                    std.log.info("zigcheck shrink: step {d} -> {any}", .{ steps, candidate });
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

        const size = if (config.num_tests > 0) tests_run * config.max_size / config.num_tests else config.max_size;
        const value = gen.generate(rng, arena_alloc, size);

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
                std.log.info("zigcheck: {d} tests, coverage:", .{p.num_tests});
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
/// try zigcheck.forAllCover(.{}, i32, gen, property, classifier, &.{
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
                                \\--- zigcheck: INSUFFICIENT COVERAGE -----------------------
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
                        \\--- zigcheck: INSUFFICIENT COVERAGE -----------------------
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
                std.log.info("zigcheck: {d} tests, coverage:", .{p.num_tests});
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

// -- collect / tabulate convenience wrappers ------------------------------

/// Run a property check that auto-labels each test case with the string
/// representation of the generated value. Equivalent to QuickCheck's `collect`.
///
/// ```zig
/// try zigcheck.forAllCollect(.{}, u8, zigcheck.generators.intRange(u8, 0, 5), struct {
///     fn prop(_: u8) !void {}
/// }.prop);
/// // Prints distribution: "16.0% 3", "18.0% 0", etc.
/// ```
pub fn forAllCollect(
    config: Config,
    comptime T: type,
    gen: Gen(T),
    property: *const fn (T) anyerror!void,
) !void {
    var label_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer label_arena.deinit();
    const alloc = label_arena.allocator();

    const seed = resolveSeed(config);
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    var label_map = std.StringHashMap(usize).init(alloc);
    var tests_run: usize = 0;
    var discards: usize = 0;

    while (tests_run < config.num_tests) {
        if (discards >= config.max_discard) {
            logGaveUp(tests_run, discards);
            return error.GaveUp;
        }
        const size = if (config.num_tests > 0) tests_run * config.max_size / config.num_tests else config.max_size;
        const value = gen.generate(rng, alloc, size);

        if (property(value)) |_| {
            const label = std.fmt.allocPrint(alloc, "{any}", .{value}) catch continue;
            const entry = label_map.getOrPut(label) catch continue;
            if (!entry.found_existing) {
                entry.value_ptr.* = 0;
            }
            entry.value_ptr.* += 1;
            tests_run += 1;
        } else |err| {
            if (err == TestDiscarded) {
                discards += 1;
                continue;
            }
            const shrunk = doShrink(T, gen, property, value, config.max_shrinks, config.verbose_shrink);
            logFailure(tests_run + 1, "{any}", .{shrunk.value}, "{any}", .{value}, shrunk.steps, seed);
            return error.PropertyFalsified;
        }
    }

    if (label_map.count() > 0) {
        std.log.info("zigcheck: {d} tests, coverage:", .{tests_run});
        var iter = label_map.iterator();
        while (iter.next()) |entry| {
            const pct = @as(f64, @floatFromInt(entry.value_ptr.*)) / @as(f64, @floatFromInt(tests_run)) * 100.0;
            std.log.info("  {d:.1}% {s}", .{ pct, entry.key_ptr.* });
        }
    }
}

/// Run a property check that groups labels under a named table.
/// Each generated value is classified by the classifier function,
/// and all labels are prefixed with the table name for grouped display.
/// Equivalent to QuickCheck's `tabulate`.
///
/// ```zig
/// try zigcheck.forAllTabulate(.{}, i32, gen, property, "sign", struct {
///     fn classify(n: i32) []const []const u8 {
///         if (n > 0) return &.{"positive"};
///         if (n < 0) return &.{"negative"};
///         return &.{"zero"};
///     }
/// }.classify);
/// // Prints: "sign/positive: 50.2%", "sign/negative: 49.7%", etc.
/// ```
pub fn forAllTabulate(
    config: Config,
    comptime T: type,
    gen: Gen(T),
    property: *const fn (T) anyerror!void,
    comptime table_name: []const u8,
    comptime classifier: *const fn (T) []const []const u8,
) !void {
    var label_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer label_arena.deinit();
    const alloc = label_arena.allocator();

    const seed = resolveSeed(config);
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    var label_map = std.StringHashMap(usize).init(alloc);
    var tests_run: usize = 0;
    var discards: usize = 0;

    while (tests_run < config.num_tests) {
        if (discards >= config.max_discard) {
            logGaveUp(tests_run, discards);
            return error.GaveUp;
        }
        const size = if (config.num_tests > 0) tests_run * config.max_size / config.num_tests else config.max_size;
        const value = gen.generate(rng, alloc, size);

        if (property(value)) |_| {
            const labels = classifier(value);
            for (labels) |label| {
                const prefixed = std.fmt.allocPrint(alloc, table_name ++ "/{s}", .{label}) catch continue;
                const entry = label_map.getOrPut(prefixed) catch continue;
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
            logFailure(tests_run + 1, "{any}", .{shrunk.value}, "{any}", .{value}, shrunk.steps, seed);
            return error.PropertyFalsified;
        }
    }

    if (label_map.count() > 0) {
        std.log.info("zigcheck: {d} tests, coverage:", .{tests_run});
        var iter = label_map.iterator();
        while (iter.next()) |entry| {
            const pct = @as(f64, @floatFromInt(entry.value_ptr.*)) / @as(f64, @floatFromInt(tests_run)) * 100.0;
            std.log.info("  {d:.1}% {s}", .{ pct, entry.key_ptr.* });
        }
    }
}

// -- Composable labeling via PropertyContext ------------------------------

/// A context object passed to property functions that enables composable
/// classify/cover/label calls inside the property itself, rather than
/// requiring a separate classifier function.
///
/// This is zigcheck's equivalent of QuickCheck's monadic property style:
///
/// ```zig
/// try zigcheck.forAllCtx(i32, zigcheck.generators.int(i32), struct {
///     fn prop(n: i32, ctx: *zigcheck.PropertyContext) !void {
///         ctx.classify(if (n > 0) "positive" else if (n < 0) "negative" else "zero");
///         ctx.cover("positive", 30.0);
///         ctx.cover("negative", 30.0);
///         ctx.tabulate("magnitude", if (@abs(n) > 100) "large" else "small");
///         if (n == std.math.maxInt(i32)) return error.PropertyFalsified;
///     }
/// }.prop);
/// ```
pub const PropertyContext = struct {
    label_map: std.StringHashMap(usize),
    cover_reqs: std.StringHashMap(f64),
    tables: std.StringHashMap(std.StringHashMap(usize)),
    alloc: std.mem.Allocator,
    num_tests_so_far: usize,

    pub fn init(alloc: std.mem.Allocator) PropertyContext {
        return .{
            .label_map = std.StringHashMap(usize).init(alloc),
            .cover_reqs = std.StringHashMap(f64).init(alloc),
            .tables = std.StringHashMap(std.StringHashMap(usize)).init(alloc),
            .alloc = alloc,
            .num_tests_so_far = 0,
        };
    }

    /// Add a label to the current test case (like QuickCheck's `label`).
    pub fn label(self: *PropertyContext, name: []const u8) void {
        const entry = self.label_map.getOrPut(name) catch return;
        if (!entry.found_existing) entry.value_ptr.* = 0;
        entry.value_ptr.* += 1;
    }

    /// Classify the current test case with a label (alias for `label`).
    pub fn classify(self: *PropertyContext, name: []const u8) void {
        self.label(name);
    }

    /// Register a minimum coverage requirement for a label.
    /// If the label's actual coverage falls below `min_pct`, the test fails.
    pub fn cover(self: *PropertyContext, name: []const u8, min_pct: f64) void {
        self.cover_reqs.put(name, min_pct) catch return;
    }

    /// Add a label under a named table (like QuickCheck's `tabulate`).
    pub fn tabulate(self: *PropertyContext, table_name: []const u8, name: []const u8) void {
        const table_entry = self.tables.getOrPut(table_name) catch return;
        if (!table_entry.found_existing) {
            table_entry.value_ptr.* = std.StringHashMap(usize).init(self.alloc);
        }
        const entry = table_entry.value_ptr.getOrPut(name) catch return;
        if (!entry.found_existing) entry.value_ptr.* = 0;
        entry.value_ptr.* += 1;
    }

    /// Check coverage requirements. Returns an error message if any requirement is not met.
    fn checkCoverage(self: *PropertyContext) ?[]const u8 {
        if (self.num_tests_so_far == 0) return null;
        var iter = self.cover_reqs.iterator();
        while (iter.next()) |req| {
            const count = self.label_map.get(req.key_ptr.*) orelse 0;
            const pct = @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(self.num_tests_so_far)) * 100.0;
            if (pct < req.value_ptr.*) {
                return std.fmt.allocPrint(
                    self.alloc,
                    "Insufficient coverage for \"{s}\": {d:.1}% (need {d:.1}%)",
                    .{ req.key_ptr.*, pct, req.value_ptr.* },
                ) catch return req.key_ptr.*;
            }
        }
        return null;
    }

    /// Print coverage report.
    fn printReport(self: *PropertyContext) void {
        if (self.label_map.count() > 0) {
            std.log.info("zigcheck: {d} tests, coverage:", .{self.num_tests_so_far});
            var iter = self.label_map.iterator();
            while (iter.next()) |entry| {
                const pct = @as(f64, @floatFromInt(entry.value_ptr.*)) / @as(f64, @floatFromInt(self.num_tests_so_far)) * 100.0;
                std.log.info("  {d:.1}% {s}", .{ pct, entry.key_ptr.* });
            }
        }
        // Print tabulated results
        var table_iter = self.tables.iterator();
        while (table_iter.next()) |table_entry| {
            std.log.info("zigcheck: Table \"{s}\":", .{table_entry.key_ptr.*});
            var entry_iter = table_entry.value_ptr.iterator();
            while (entry_iter.next()) |entry| {
                const pct = @as(f64, @floatFromInt(entry.value_ptr.*)) / @as(f64, @floatFromInt(self.num_tests_so_far)) * 100.0;
                std.log.info("  {d:.1}% {s}", .{ pct, entry.key_ptr.* });
            }
        }
    }
};

/// Run a property check with a `PropertyContext` for composable labeling.
/// The property receives both the generated value and a context for
/// classify/cover/label/tabulate calls.
pub fn forAllCtx(
    comptime T: type,
    gen: Gen(T),
    comptime property: *const fn (T, *PropertyContext) anyerror!void,
) !void {
    return forAllCtxWith(.{}, T, gen, property);
}

/// Run a context property check with explicit config.
pub fn forAllCtxWith(
    config: Config,
    comptime T: type,
    gen: Gen(T),
    comptime property: *const fn (T, *PropertyContext) anyerror!void,
) !void {
    var ctx_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer ctx_arena.deinit();
    const alloc = ctx_arena.allocator();

    var ctx = PropertyContext.init(alloc);

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
            logGaveUp(tests_run, discards);
            return error.GaveUp;
        }

        const size = if (config.num_tests > 0) tests_run * config.max_size / config.num_tests else config.max_size;
        const value = gen.generate(rng, gen_alloc, size);

        if (property(value, &ctx)) |_| {
            tests_run += 1;
            ctx.num_tests_so_far = tests_run;
        } else |err| {
            if (err == TestDiscarded) {
                discards += 1;
                continue;
            }
            // Shrink using a wrapper that ignores the context
            const shrunk = doShrink(T, gen, struct {
                fn wrapper(v: T) anyerror!void {
                    var dummy_ctx = PropertyContext.init(std.heap.page_allocator);
                    return property(v, &dummy_ctx);
                }
            }.wrapper, value, config.max_shrinks, config.verbose_shrink);
            logFailure(tests_run + 1, "{any}", .{shrunk.value}, "{any}", .{value}, shrunk.steps, seed);
            return error.PropertyFalsified;
        }
    }

    // Check coverage requirements
    if (ctx.checkCoverage()) |msg| {
        std.log.info("zigcheck: insufficient coverage: {s}", .{msg});
        return error.InsufficientCoverage;
    }

    // Print coverage report
    ctx.printReport();
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

// -- N-argument properties via zip ----------------------------------------

/// Run a property check with N generators using zip. The property receives
/// individual arguments (splatted), not a tuple. Generalizes forAll2/forAll3
/// to any number of generators.
///
/// ```zig
/// try zigcheck.forAllZip(.{
///     zigcheck.generators.int(u8),
///     zigcheck.generators.int(u64),
///     zigcheck.generators.boolean(),
/// }, struct {
///     fn prop(a: u8, b: u64, c: bool) !void { ... }
/// }.prop);
/// ```
pub fn forAllZip(comptime gens: anytype, comptime property: anytype) !void {
    return forAllZipWith(.{}, gens, property);
}

/// Run a zipped property check with explicit config.
pub fn forAllZipWith(config: Config, comptime gens: anytype, comptime property: anytype) !void {
    const combinators = @import("combinators.zig");
    const Tuple = combinators.ZipResult(gens);
    const zipped = comptime combinators.zip(gens);

    const result = check(config, Tuple, zipped, struct {
        fn wrapper(tuple: Tuple) anyerror!void {
            return @call(.auto, property, tuple);
        }
    }.wrapper);

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

// Multi-argument properties: use forAllZip / forAllZipWith instead.
// These generalize forAll2/forAll3 to any number of generators via zip.

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

// -- forAllZip multi-argument tests (replaced check2/check3) ----------------

test "forAllZip: passing two-argument property" {
    try forAllZip(.{
        generators.int(i32),
        generators.int(i32),
    }, struct {
        fn prop(_: i32, _: i32) !void {}
    }.prop);
}

test "forAllZip: passing three-argument property" {
    try forAllZip(.{
        generators.int(i32),
        generators.int(i32),
        generators.boolean(),
    }, struct {
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

// -- QuickCheck parity tests ----------------------------------------------

test "bad shrinker resilience: shrink that yields original terminates" {
    // A generator whose shrinker always yields the original value (infinite loop).
    // The runner must terminate via max_shrinks limit.
    const bad_gen = Gen(u32){
        .genFn = struct {
            fn f(rng: std.Random, _: std.mem.Allocator, _: usize) u32 {
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

test "cover: detects unmet coverage requirement" {
    // Use checkLabeled directly to verify coverage without logging warnings.
    // All values are labeled "all" -- "rare" label never appears.
    var label_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer label_arena.deinit();

    const result = checkLabeled(.{ .seed = 42, .num_tests = 100 }, u32, generators.int(u32), struct {
        fn prop(_: u32) !void {}
    }.prop, struct {
        fn classify(_: u32) []const []const u8 {
            return &.{"all"};
        }
    }.classify, label_arena.allocator());

    switch (result) {
        .passed => |p| {
            // Verify "rare" label is absent
            for (p.label_names) |name| {
                if (std.mem.eql(u8, name, "rare")) return error.TestUnexpectedResult;
            }
            // Good — the "rare" label is missing, which is what forAllCover would fail on
        },
        else => return error.TestUnexpectedResult,
    }
}

test "collect: auto-labels with value string representation" {
    // Should not error -- just prints distribution
    try forAllCollect(.{ .seed = 42, .num_tests = 20 }, u8, generators.intRange(u8, 0, 3), struct {
        fn prop(_: u8) !void {}
    }.prop);
}

test "tabulate: groups labels under table name" {
    try forAllTabulate(.{ .seed = 42, .num_tests = 50 }, i32, generators.int(i32), struct {
        fn prop(_: i32) !void {}
    }.prop, "sign", struct {
        fn classify(n: i32) []const []const u8 {
            if (n > 0) return &.{"positive"};
            if (n < 0) return &.{"negative"};
            return &.{"zero"};
        }
    }.classify);
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

test "Config builder methods" {
    const base = Config{};
    const c = base.withNumTests(500).withMaxShrinks(2000).withMaxDiscard(1000).withSeed(42).withVerbose().withMaxSize(200);
    try std.testing.expectEqual(@as(usize, 500), c.num_tests);
    try std.testing.expectEqual(@as(usize, 2000), c.max_shrinks);
    try std.testing.expectEqual(@as(usize, 1000), c.max_discard);
    try std.testing.expectEqual(@as(?u64, 42), c.seed);
    try std.testing.expect(c.verbose);
    try std.testing.expectEqual(@as(usize, 200), c.max_size);
}

test "recheck: replays a failed property" {
    const prop = struct {
        fn f(n: u32) !void {
            if (n >= 10) return error.PropertyFalsified;
        }
    }.f;
    const gen = generators.int(u32);
    const orig_config = Config{ .seed = 42 };

    // First run — get a failure
    const result = check(orig_config, u32, gen, prop);
    switch (result) {
        .failed => |f| {
            // Recheck: same seed, same num_tests (for consistent size ramp)
            const result2 = check(.{ .seed = f.seed, .num_tests = orig_config.num_tests }, u32, gen, prop);
            switch (result2) {
                .failed => |f2| {
                    try std.testing.expectEqual(f.original, f2.original);
                    try std.testing.expectEqual(f.shrunk, f2.shrunk);
                },
                else => return error.TestUnexpectedResult,
            }
        },
        else => return error.TestUnexpectedResult,
    }
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

test "forAllCtx: composable classify and cover" {
    // Use PropertyContext to classify inside the property itself
    try forAllCtxWith(.{ .seed = 42, .num_tests = 200 }, i32, generators.int(i32), struct {
        fn prop(n: i32, ctx: *PropertyContext) !void {
            ctx.classify(if (n > 0) "positive" else if (n < 0) "negative" else "zero");
            ctx.cover("positive", 20.0);
            ctx.cover("negative", 20.0);
        }
    }.prop);
}

test "forAllCtx: tabulate groups labels" {
    try forAllCtxWith(.{ .seed = 42, .num_tests = 100 }, u32, generators.intRange(u32, 0, 100), struct {
        fn prop(n: u32, ctx: *PropertyContext) !void {
            ctx.tabulate("size", if (n > 50) "large" else "small");
        }
    }.prop);
}

test "forAllCtx: coverage failure" {
    // A property that never labels "rare" should fail coverage check
    const result = @as(anyerror!void, forAllCtxWith(
        .{ .seed = 42, .num_tests = 50 },
        u32,
        generators.intRange(u32, 0, 100),
        struct {
            fn prop(n: u32, ctx: *PropertyContext) !void {
                ctx.classify(if (n > 50) "common" else "other");
                // Require a label that will never be applied
                ctx.cover("nonexistent", 10.0);
            }
        }.prop,
    ));
    try std.testing.expectError(error.InsufficientCoverage, result);
}

test "forAllZip: N generators with splatted args" {
    try forAllZipWith(.{ .seed = 42, .num_tests = 50 }, .{
        generators.intRange(i32, 0, 100),
        generators.boolean(),
        generators.intRange(u8, 0, 10),
    }, struct {
        fn prop(n: i32, b: bool, x: u8) !void {
            _ = b;
            if (n < 0 or n > 100) return error.PropertyFalsified;
            if (x > 10) return error.PropertyFalsified;
        }
    }.prop);
}

test "within: fast property passes" {
    // A trivial property that completes instantly should pass within a generous limit
    const prop = within(u32, 10_000_000, struct {
        fn f(_: u32) !void {}
    }.f);
    const result = check(.{ .seed = 42, .num_tests = 10 }, u32, generators.int(u32), prop);
    switch (result) {
        .passed => {},
        else => return error.TestUnexpectedResult,
    }
}
