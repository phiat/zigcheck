// Fuzz integration -- coverage-guided property testing via std.testing.fuzz.
//
// FuzzRandom adapts a []const u8 byte slice (from libFuzzer) into a
// std.Random interface. Generators consume it exactly as they consume a
// PRNG, but the bytes are chosen by coverage feedback instead of a seed.
// When the slice is exhausted, zeros are returned -- producing degenerate
// but valid values without crashing.

const std = @import("std");
const Gen = @import("gen.zig").Gen;

/// A std.Random backed by a fixed byte slice instead of a PRNG.
/// Drains bytes sequentially; returns zeros when exhausted.
const FuzzRandom = struct {
    bytes: []const u8,
    pos: usize,

    pub fn random(self: *FuzzRandom) std.Random {
        return std.Random.init(self, fill);
    }

    fn fill(self: *FuzzRandom, buf: []u8) void {
        for (buf) |*byte| {
            byte.* = if (self.pos < self.bytes.len) blk: {
                defer self.pos += 1;
                break :blk self.bytes[self.pos];
            } else 0;
        }
    }
};

/// Generate one structured value of type T from raw fuzz bytes.
///
/// The bytes drive the generator's randomness. If bytes are exhausted
/// mid-generation, remaining random draws return zero -- producing a
/// degenerate but valid value. Size is fixed at 100 (max).
pub fn fromFuzzInput(comptime T: type, gen: Gen(T), bytes: []const u8, allocator: std.mem.Allocator) T {
    var fuzz_rng = FuzzRandom{ .bytes = bytes, .pos = 0 };
    return gen.generate(fuzz_rng.random(), allocator, 100);
}

// -- Tests ----------------------------------------------------------------

const testing = std.testing;
const runner = @import("runner.zig");
const generators = @import("generators.zig");

test "FuzzRandom: drains bytes in order" {
    const bytes = [_]u8{ 0xAA, 0xBB, 0xCC, 0xDD };
    var fuzz_rng = FuzzRandom{ .bytes = &bytes, .pos = 0 };
    const rng = fuzz_rng.random();

    var buf: [4]u8 = undefined;
    rng.bytes(&buf);
    try testing.expectEqualSlices(u8, &bytes, &buf);
}

test "FuzzRandom: fills zeros when exhausted" {
    const bytes = [_]u8{ 0x42, 0x43 };
    var fuzz_rng = FuzzRandom{ .bytes = &bytes, .pos = 0 };
    const rng = fuzz_rng.random();

    var buf: [5]u8 = undefined;
    rng.bytes(&buf);
    try testing.expectEqualSlices(u8, &[_]u8{ 0x42, 0x43, 0x00, 0x00, 0x00 }, &buf);
}

test "FuzzRandom: empty bytes gives all zeros" {
    var fuzz_rng = FuzzRandom{ .bytes = &[_]u8{}, .pos = 0 };
    const rng = fuzz_rng.random();

    var buf: [4]u8 = undefined;
    rng.bytes(&buf);
    try testing.expectEqualSlices(u8, &[_]u8{ 0, 0, 0, 0 }, &buf);
}

test "fromFuzzInput: deterministic — same bytes produce same value" {
    const gen = generators.auto(u32);
    const bytes = [_]u8{ 0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04 };

    const a = fromFuzzInput(u32, gen, &bytes, testing.allocator);
    const b = fromFuzzInput(u32, gen, &bytes, testing.allocator);
    try testing.expectEqual(a, b);
}

test "fromFuzzInput: empty bytes produces valid value" {
    const gen = generators.auto(u32);
    // Should not crash — zeros are valid random input
    _ = fromFuzzInput(u32, gen, &[_]u8{}, testing.allocator);
}

test "checkFuzzOne: failing property returns .failed with shrunk value" {
    const gen = generators.auto(u8);
    const bytes = [_]u8{ 0x05, 0x0A, 0x0F, 0x14 };

    const result = runner.checkFuzzOne(u8, gen, &bytes, struct {
        fn prop(_: u8) anyerror!void {
            return error.PropertyFalsified;
        }
    }.prop);

    switch (result) {
        .failed => |f| {
            // Shrinking an always-failing u8 property should reach 0
            try testing.expectEqual(@as(u8, 0), f.shrunk);
            try testing.expect(f.shrink_steps > 0);
        },
        .passed => return error.TestUnexpectedResult,
    }
}

test "checkFuzzOne: passing property returns .passed" {
    const gen = generators.auto(u8);
    const bytes = [_]u8{ 0x05 };

    const result = runner.checkFuzzOne(u8, gen, &bytes, struct {
        fn prop(_: u8) anyerror!void {}
    }.prop);

    switch (result) {
        .passed => {},
        .failed => return error.TestUnexpectedResult,
    }
}

test "checkFuzzOne: discarded test cases count as passed" {
    const gen = generators.auto(u8);
    const bytes = [_]u8{ 0x05 };

    const result = runner.checkFuzzOne(u8, gen, &bytes, struct {
        fn prop(_: u8) anyerror!void {
            return error.TestDiscarded;
        }
    }.prop);

    switch (result) {
        .passed => {},
        .failed => return error.TestUnexpectedResult,
    }
}

test "forAllFuzzOne: passing property returns success" {
    const gen = generators.auto(u8);
    const bytes = [_]u8{ 0x05 };

    try runner.forAllFuzzOne(u8, gen, &bytes, struct {
        fn prop(_: u8) anyerror!void {}
    }.prop);
}

test "forAllFuzzOne: discarded test cases are not failures" {
    const gen = generators.auto(u8);
    const bytes = [_]u8{ 0x05 };

    try runner.forAllFuzzOne(u8, gen, &bytes, struct {
        fn prop(_: u8) anyerror!void {
            return error.TestDiscarded;
        }
    }.prop);
}
