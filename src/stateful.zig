// Stateful / state-machine testing for sequential API verification.
//
// Users define a model, a command type, and transitions. The framework
// generates random command sequences, runs them against both model and
// system-under-test, and shrinks failing sequences to minimal counterexamples.
//
// Inspired by QuickCheck's Test.QuickCheck.Monadic and Erlang QuickCheck's
// eqc_statem, adapted to Zig's constraints (no closures, comptime generics).

const std = @import("std");
const Gen = @import("gen.zig").Gen;
const ShrinkIter = @import("shrink.zig").ShrinkIter;

/// Configuration for stateful tests.
pub const StatefulConfig = struct {
    /// Maximum number of test sequences to run.
    num_tests: usize = 100,
    /// Maximum number of commands per sequence.
    max_commands: usize = 50,
    /// Maximum shrink iterations.
    max_shrinks: usize = 1000,
    /// Fixed seed for reproducibility (null = time-based).
    seed: ?u64 = null,
    /// Print each command during execution.
    verbose: bool = false,
};

/// Define a state machine test specification.
///
/// `Cmd` is the command tagged union type.
/// `Model` is the model state type.
/// `Sut` is the system-under-test type (the real implementation).
///
/// Example:
/// ```zig
/// const Spec = zigcheck.StateMachine(Command, Model, *MyStack);
/// ```
pub fn StateMachine(
    comptime Cmd: type,
    comptime Model: type,
    comptime Sut: type,
) type {
    return struct {
        const Self = @This();

        /// User-provided callbacks bundled into a vtable-like struct.
        pub const Callbacks = struct {
            /// Generate a random command given the current model state.
            gen_command: *const fn (Model, std.Random, std.mem.Allocator) ?Cmd,
            /// Apply a command to the model, returning the next model state.
            next_model: *const fn (Model, Cmd) Model,
            /// Check if a command is valid in the current model state.
            precondition: *const fn (Model, Cmd) bool,
            /// Execute the command on the real system. Returns error on failure.
            run_command: *const fn (Sut, Cmd) anyerror!void,
            /// Check postcondition after running a command.
            /// Receives model-before, command, and the SUT.
            postcondition: *const fn (Model, Cmd, Sut) anyerror!void,
            /// Create initial model state.
            init_model: *const fn () Model,
            /// Create initial SUT state.
            init_sut: *const fn (std.mem.Allocator) anyerror!Sut,
            /// Clean up SUT state (optional).
            cleanup_sut: ?*const fn (Sut) void = null,
        };

        /// Result of a stateful test run.
        pub const Result = union(enum) {
            passed: struct {
                num_tests: usize,
            },
            failed: struct {
                seed: u64,
                sequence: []const Cmd,
                shrunk_sequence: []const Cmd,
                shrink_steps: usize,
                failing_step: usize,
                num_tests_before_fail: usize,
            },
        };

        /// Run the stateful test with default config.
        pub fn run(callbacks: Callbacks) !void {
            return runWith(.{}, callbacks);
        }

        /// Run the stateful test with explicit config.
        pub fn runWith(config: StatefulConfig, callbacks: Callbacks) !void {
            const result = check(config, callbacks);
            switch (result) {
                .passed => {},
                .failed => |f| {
                    std.log.err(
                        \\
                        \\--- zigcheck: STATEFUL TEST FAILED ---
                        \\Seed: 0x{x}
                        \\Failed after {d} tests, at step {d}
                        \\Original sequence ({d} commands):
                    , .{ f.seed, f.num_tests_before_fail, f.failing_step, f.sequence.len });
                    for (f.sequence, 0..) |cmd, i| {
                        std.log.err("  [{d}] {any}", .{ i, cmd });
                    }
                    std.log.err("Shrunk sequence ({d} commands, {d} shrink steps):", .{ f.shrunk_sequence.len, f.shrink_steps });
                    for (f.shrunk_sequence, 0..) |cmd, i| {
                        std.log.err("  [{d}] {any}", .{ i, cmd });
                    }
                    return error.StatefulTestFailed;
                },
            }
        }

        /// Run the stateful test and return the result without failing.
        pub fn check(config: StatefulConfig, callbacks: Callbacks) Result {
            const seed = config.seed orelse @as(u64, @intCast(@as(u128, @bitCast(std.time.nanoTimestamp())) & 0xFFFFFFFFFFFFFFFF));
            var prng = std.Random.DefaultPrng.init(seed);
            const rng = prng.random();

            var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
            defer arena_state.deinit();
            const alloc = arena_state.allocator();

            for (0..config.num_tests) |test_idx| {
                // Generate a random command sequence
                const max_len = @min(config.max_commands, test_idx * config.max_commands / @max(config.num_tests, 1) + 2);
                const seq_len = if (max_len <= 1) 1 else rng.intRangeAtMost(usize, 1, max_len);

                var model = callbacks.init_model();
                var sequence = alloc.alloc(Cmd, seq_len) catch continue;
                var actual_len: usize = 0;

                // Generate commands respecting preconditions
                for (0..seq_len) |_| {
                    if (callbacks.gen_command(model, rng, alloc)) |cmd| {
                        if (callbacks.precondition(model, cmd)) {
                            sequence[actual_len] = cmd;
                            actual_len += 1;
                            model = callbacks.next_model(model, cmd);
                        }
                    }
                }
                sequence = sequence[0..actual_len];

                if (sequence.len == 0) continue;

                // Execute the sequence and check postconditions
                if (executeSequence(callbacks, sequence, alloc, config.verbose)) |failing_step| {
                    // Shrink the sequence
                    const shrunk = shrinkSequence(config, callbacks, sequence, alloc);
                    return .{ .failed = .{
                        .seed = seed,
                        .sequence = sequence,
                        .shrunk_sequence = shrunk.sequence,
                        .shrink_steps = shrunk.steps,
                        .failing_step = failing_step,
                        .num_tests_before_fail = test_idx + 1,
                    } };
                }
            }

            return .{ .passed = .{ .num_tests = config.num_tests } };
        }

        /// Execute a command sequence, returning the failing step index or null if all pass.
        fn executeSequence(
            callbacks: Callbacks,
            sequence: []const Cmd,
            alloc: std.mem.Allocator,
            verbose: bool,
        ) ?usize {
            var model = callbacks.init_model();
            const sut = callbacks.init_sut(alloc) catch return 0;
            defer if (callbacks.cleanup_sut) |cleanup| cleanup(sut);

            for (sequence, 0..) |cmd, i| {
                if (verbose) {
                    std.log.info("zigcheck: step [{d}] {any}", .{ i, cmd });
                }

                // Run the command on the SUT
                callbacks.run_command(sut, cmd) catch return i;

                // Check postcondition
                callbacks.postcondition(model, cmd, sut) catch return i;

                // Advance model
                model = callbacks.next_model(model, cmd);
            }
            return null;
        }

        const ShrunkResult = struct {
            sequence: []const Cmd,
            steps: usize,
        };

        /// Shrink a failing sequence by trying to remove commands.
        fn shrinkSequence(
            config: StatefulConfig,
            callbacks: Callbacks,
            original: []const Cmd,
            alloc: std.mem.Allocator,
        ) ShrunkResult {
            var best = alloc.dupe(Cmd, original) catch return .{ .sequence = original, .steps = 0 };
            var steps: usize = 0;

            // Try removing chunks of decreasing size (same strategy as slice shrinking)
            var chunk_size = best.len / 2;
            while (chunk_size >= 1 and steps < config.max_shrinks) : (chunk_size /= 2) {
                var start: usize = 0;
                while (start + chunk_size <= best.len and steps < config.max_shrinks) {
                    steps += 1;
                    // Create candidate with chunk removed
                    const candidate = alloc.alloc(Cmd, best.len - chunk_size) catch break;
                    @memcpy(candidate[0..start], best[0..start]);
                    @memcpy(candidate[start..], best[start + chunk_size ..]);

                    // Check preconditions still hold for the shortened sequence
                    if (isValidSequence(callbacks, candidate) and
                        executeSequence(callbacks, candidate, alloc, false) != null)
                    {
                        best = candidate;
                        // Don't advance start — try removing from same position again
                    } else {
                        start += 1;
                    }
                }
                if (chunk_size == 1) break;
            }

            return .{ .sequence = best, .steps = steps };
        }

        /// Check that all preconditions hold for a sequence.
        fn isValidSequence(callbacks: Callbacks, sequence: []const Cmd) bool {
            var model = callbacks.init_model();
            for (sequence) |cmd| {
                if (!callbacks.precondition(model, cmd)) return false;
                model = callbacks.next_model(model, cmd);
            }
            return true;
        }
    };
}

// -- Tests ----------------------------------------------------------------

const testing = std.testing;

/// A simple stack for testing the stateful framework.
const TestStack = struct {
    items: [64]i32 = undefined,
    len: usize = 0,

    fn push(self: *TestStack, val: i32) void {
        if (self.len < 64) {
            self.items[self.len] = val;
            self.len += 1;
        }
    }

    fn pop(self: *TestStack) ?i32 {
        if (self.len == 0) return null;
        self.len -= 1;
        return self.items[self.len];
    }

    fn size(self: *const TestStack) usize {
        return self.len;
    }
};

const StackCmd = union(enum) {
    push: i32,
    pop,
};

const StackModel = struct {
    size: usize = 0,
};

test "stateful: correct stack passes" {
    const Spec = StateMachine(StackCmd, StackModel, *TestStack);

    const result = Spec.check(.{ .seed = 42, .num_tests = 50, .max_commands = 20 }, .{
        .init_model = struct {
            fn f() StackModel {
                return .{};
            }
        }.f,
        .init_sut = struct {
            fn f(alloc: std.mem.Allocator) !*TestStack {
                const stack = try alloc.create(TestStack);
                stack.* = .{};
                return stack;
            }
        }.f,
        .gen_command = struct {
            fn f(_: StackModel, rng: std.Random, _: std.mem.Allocator) ?StackCmd {
                if (rng.boolean()) {
                    return .{ .push = rng.intRangeAtMost(i32, -10, 10) };
                } else {
                    return .pop;
                }
            }
        }.f,
        .precondition = struct {
            fn f(model: StackModel, cmd: StackCmd) bool {
                return switch (cmd) {
                    .push => model.size < 64,
                    .pop => model.size > 0,
                };
            }
        }.f,
        .run_command = struct {
            fn f(sut: *TestStack, cmd: StackCmd) !void {
                switch (cmd) {
                    .push => |val| sut.push(val),
                    .pop => _ = sut.pop(),
                }
            }
        }.f,
        .next_model = struct {
            fn f(model: StackModel, cmd: StackCmd) StackModel {
                return switch (cmd) {
                    .push => .{ .size = model.size + 1 },
                    .pop => .{ .size = model.size - 1 },
                };
            }
        }.f,
        .postcondition = struct {
            fn f(model: StackModel, cmd: StackCmd, sut: *TestStack) !void {
                const expected_size: usize = switch (cmd) {
                    .push => model.size + 1,
                    .pop => model.size - 1,
                };
                if (sut.size() != expected_size) {
                    return error.SizeMismatch;
                }
            }
        }.f,
    });

    switch (result) {
        .passed => {},
        .failed => return error.TestUnexpectedResult,
    }
}

test "stateful: buggy stack detects failure and shrinks" {
    // A "buggy" stack that breaks when size exceeds 3
    const BuggyStack = struct {
        items: [64]i32 = undefined,
        len: usize = 0,

        fn push(self: *@This(), val: i32) void {
            if (self.len < 64) {
                self.items[self.len] = val;
                self.len += 1;
            }
        }

        fn pop(self: *@This()) ?i32 {
            if (self.len == 0) return null;
            self.len -= 1;
            return self.items[self.len];
        }

        fn size(self: *const @This()) usize {
            // Bug: returns wrong size when > 3
            if (self.len > 3) return 0;
            return self.len;
        }
    };

    const Spec = StateMachine(StackCmd, StackModel, *BuggyStack);

    const result = Spec.check(.{ .seed = 42, .num_tests = 50, .max_commands = 20 }, .{
        .init_model = struct {
            fn f() StackModel {
                return .{};
            }
        }.f,
        .init_sut = struct {
            fn f(alloc: std.mem.Allocator) !*BuggyStack {
                const stack = try alloc.create(BuggyStack);
                stack.* = .{};
                return stack;
            }
        }.f,
        .gen_command = struct {
            fn f(_: StackModel, rng: std.Random, _: std.mem.Allocator) ?StackCmd {
                if (rng.boolean()) {
                    return .{ .push = rng.intRangeAtMost(i32, -10, 10) };
                } else {
                    return .pop;
                }
            }
        }.f,
        .precondition = struct {
            fn f(model: StackModel, cmd: StackCmd) bool {
                return switch (cmd) {
                    .push => model.size < 64,
                    .pop => model.size > 0,
                };
            }
        }.f,
        .run_command = struct {
            fn f(sut: *BuggyStack, cmd: StackCmd) !void {
                switch (cmd) {
                    .push => |val| sut.push(val),
                    .pop => _ = sut.pop(),
                }
            }
        }.f,
        .next_model = struct {
            fn f(model: StackModel, cmd: StackCmd) StackModel {
                return switch (cmd) {
                    .push => .{ .size = model.size + 1 },
                    .pop => .{ .size = model.size - 1 },
                };
            }
        }.f,
        .postcondition = struct {
            fn f(model: StackModel, cmd: StackCmd, sut: *BuggyStack) !void {
                const expected_size: usize = switch (cmd) {
                    .push => model.size + 1,
                    .pop => model.size - 1,
                };
                if (sut.size() != expected_size) {
                    return error.SizeMismatch;
                }
            }
        }.f,
    });

    switch (result) {
        .failed => |f| {
            // The shrunk sequence should be minimal: exactly 4 pushes to trigger the bug
            try testing.expect(f.shrunk_sequence.len <= f.sequence.len);
            try testing.expect(f.shrunk_sequence.len >= 4); // Need >3 pushes to trigger bug
        },
        .passed => return error.TestUnexpectedResult,
    }
}
