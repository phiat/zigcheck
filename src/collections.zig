// Collection generators: slices, strings, unicode, shuffle, sublist, ordered lists.

const std = @import("std");
const Gen = @import("gen.zig").Gen;
const ShrinkIter = @import("shrink.zig").ShrinkIter;
const shrink = @import("shrink.zig");
const generators = @import("generators.zig");
const combinators = @import("combinators.zig");

// -- Slice and string generators ------------------------------------------

/// Generator for a single printable ASCII character (32-126).
pub fn asciiChar() Gen(u8) {
    return generators.intRange(u8, 32, 126);
}

/// Generator for a single alphanumeric character [a-zA-Z0-9].
pub fn alphanumeric() Gen(u8) {
    const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    return combinators.element(u8, charset);
}

/// Generator for slices of T with length in [0, max_len].
pub fn slice(comptime T: type, comptime inner: Gen(T), comptime max_len: usize) Gen([]const T) {
    return sliceRange(T, inner, 0, max_len);
}

/// Generator for slices of T with length in [min_len, max_len].
pub fn sliceRange(comptime T: type, comptime inner: Gen(T), comptime min_len: usize, comptime max_len: usize) Gen([]const T) {
    comptime {
        if (min_len > max_len) @compileError("sliceRange(): min_len must be <= max_len");
    }
    return .{
        .genFn = struct {
            fn gen(rng: std.Random, allocator: std.mem.Allocator, size: usize) []const T {
                const scaled_max = @min(min_len + (max_len - min_len) * size / 100, max_len);
                const len = rng.intRangeAtMost(usize, min_len, scaled_max);
                const result = allocator.alloc(T, len) catch return &.{};
                for (result) |*slot| {
                    slot.* = inner.generate(rng, allocator, size);
                }
                return result;
            }
        }.gen,
        .shrinkFn = struct {
            fn shrinkFn(value: []const T, allocator: std.mem.Allocator) ShrinkIter([]const T) {
                if (value.len <= min_len) return ShrinkIter([]const T).empty();

                // Shrink phases (QuickCheck-style):
                //  1. Remove chunks: halves, quarters, eighths, ... down to single deletions
                //  2. Shrink individual elements in place
                const Phase = enum { chunks, elements };

                const State = struct {
                    original: []const T,
                    alloc: std.mem.Allocator,
                    phase: Phase,
                    // Chunk-removal phase: current chunk size and start offset
                    chunk_size: usize,
                    chunk_start: usize,
                    // Element-shrink phase: position and current element's shrinker
                    elem_idx: usize,
                    elem_shrinker: ShrinkIter(T),

                    fn nextImpl(self: *@This()) ?[]const T {
                        switch (self.phase) {
                            .chunks => {
                                if (self.tryRemoveChunk()) |result| return result;
                                self.phase = .elements;
                                self.elem_idx = 0;
                                self.startElementShrinker();
                                return self.nextImpl();
                            },
                            .elements => {
                                return self.tryShrinkElement();
                            },
                        }
                    }

                    /// Try removing a chunk of current size at current position.
                    /// When all positions exhausted, halve the chunk size.
                    /// Stop when chunk size reaches 0.
                    fn tryRemoveChunk(self: *@This()) ?[]const T {
                        while (self.chunk_size > 0) {
                            while (self.chunk_start + self.chunk_size <= self.original.len) {
                                const start = self.chunk_start;
                                const end = start + self.chunk_size;
                                self.chunk_start += 1;
                                const new_len = self.original.len - self.chunk_size;
                                if (new_len < min_len) continue;
                                return self.copyWithoutRange(start, end);
                            }
                            // Halve chunk size, reset start
                            self.chunk_size /= 2;
                            self.chunk_start = 0;
                        }
                        return null;
                    }

                    /// Try shrinking the current element, advancing to the next when exhausted.
                    fn tryShrinkElement(self: *@This()) ?[]const T {
                        while (self.elem_idx < self.original.len) {
                            if (self.elem_shrinker.next()) |shrunk_val| {
                                return self.copyWithElement(self.elem_idx, shrunk_val);
                            }
                            self.elem_idx += 1;
                            self.startElementShrinker();
                        }
                        return null;
                    }

                    /// Copy original with range [start, end) removed.
                    fn copyWithoutRange(self: *@This(), start: usize, end: usize) ?[]const T {
                        const removed = end - start;
                        const new_len = self.original.len - removed;
                        const result = self.alloc.alloc(T, new_len) catch return null;
                        if (start > 0) @memcpy(result[0..start], self.original[0..start]);
                        if (end < self.original.len) @memcpy(result[start..new_len], self.original[end..self.original.len]);
                        return result;
                    }

                    fn copyWithElement(self: *@This(), idx: usize, val: T) ?[]const T {
                        const result = self.alloc.alloc(T, self.original.len) catch return null;
                        @memcpy(result, self.original);
                        result[idx] = val;
                        return result;
                    }

                    fn startElementShrinker(self: *@This()) void {
                        if (self.elem_idx < self.original.len) {
                            self.elem_shrinker = inner.shrink(self.original[self.elem_idx], self.alloc);
                        }
                    }

                    fn typeErasedNext(ctx: *anyopaque) ?[]const T {
                        const self: *@This() = @ptrCast(@alignCast(ctx));
                        return self.nextImpl();
                    }
                };

                // Start with chunk_size = half of original length
                const initial_chunk = (value.len - min_len) / 2;
                const effective_chunk = if (initial_chunk == 0) 1 else initial_chunk;

                const state = allocator.create(State) catch return ShrinkIter([]const T).empty();
                state.* = .{
                    .original = value,
                    .alloc = allocator,
                    .phase = .chunks,
                    .chunk_size = effective_chunk,
                    .chunk_start = 0,
                    .elem_idx = 0,
                    .elem_shrinker = ShrinkIter(T).empty(),
                };

                return .{
                    .context = @ptrCast(state),
                    .nextFn = State.typeErasedNext,
                };
            }
        }.shrinkFn,
    };
}

/// Generator for ASCII strings (printable characters, 32-126) up to max_len.
pub fn asciiString(comptime max_len: usize) Gen([]const u8) {
    return slice(u8, asciiChar(), max_len);
}

/// Generator for ASCII strings with length in [min_len, max_len].
pub fn asciiStringRange(comptime min_len: usize, comptime max_len: usize) Gen([]const u8) {
    return sliceRange(u8, asciiChar(), min_len, max_len);
}

/// Generator for alphanumeric strings [a-zA-Z0-9] up to max_len.
pub fn alphanumericString(comptime max_len: usize) Gen([]const u8) {
    return slice(u8, alphanumeric(), max_len);
}

/// Generator for raw byte strings (any u8 value) up to max_len.
pub fn string(comptime max_len: usize) Gen([]const u8) {
    return slice(u8, generators.int(u8), max_len);
}

/// Generator for a random Unicode code point (excludes surrogates).
pub fn unicodeChar() Gen(u21) {
    return .{
        .genFn = struct {
            fn gen(rng: std.Random, _: std.mem.Allocator, _: usize) u21 {
                while (true) {
                    const cp = rng.intRangeAtMost(u21, 0, 0x10FFFF);
                    // Exclude surrogates (U+D800..U+DFFF)
                    if (cp >= 0xD800 and cp <= 0xDFFF) continue;
                    return cp;
                }
            }
        }.gen,
        .shrinkFn = struct {
            fn f(value: u21, allocator: std.mem.Allocator) ShrinkIter(u21) {
                // Wrap IntShrinkState to skip surrogate code points (U+D800..U+DFFF)
                const State = struct {
                    inner: shrink.IntShrinkState(u21),

                    fn nextValid(self: *@This()) ?u21 {
                        while (self.inner.next()) |cp| {
                            if (cp >= 0xD800 and cp <= 0xDFFF) continue;
                            return cp;
                        }
                        return null;
                    }

                    fn typeErasedNext(ctx: *anyopaque) ?u21 {
                        const self: *@This() = @ptrCast(@alignCast(ctx));
                        return self.nextValid();
                    }
                };
                const state = allocator.create(State) catch return ShrinkIter(u21).empty();
                state.* = .{ .inner = shrink.IntShrinkState(u21).init(value) };
                return .{
                    .context = @ptrCast(state),
                    .nextFn = State.typeErasedNext,
                };
            }
        }.f,
    };
}

/// Generator for random Unicode strings (valid UTF-8) up to max_codepoints.
///
/// Shrinking tries shorter prefixes only (not individual codepoint simplification).
/// This preserves UTF-8 validity but is less thorough than ASCII string shrinking.
pub fn unicodeString(comptime max_codepoints: usize) Gen([]const u8) {
    return .{
        .genFn = struct {
            fn gen(rng: std.Random, allocator: std.mem.Allocator, _: usize) []const u8 {
                const num_cps = rng.intRangeAtMost(usize, 0, max_codepoints);
                // Worst case: 4 bytes per codepoint
                var buf = allocator.alloc(u8, num_cps * 4) catch return "";
                var pos: usize = 0;
                for (0..num_cps) |_| {
                    const cp: u21 = blk: {
                        while (true) {
                            const c = rng.intRangeAtMost(u21, 0, 0x10FFFF);
                            if (c >= 0xD800 and c <= 0xDFFF) continue;
                            break :blk c;
                        }
                    };
                    const len = std.unicode.utf8CodepointSequenceLength(cp) catch continue;
                    if (pos + len > buf.len) break;
                    _ = std.unicode.utf8Encode(cp, buf[pos..]) catch continue;
                    pos += len;
                }
                return buf[0..pos];
            }
        }.gen,
        .shrinkFn = struct {
            fn f(value: []const u8, allocator: std.mem.Allocator) ShrinkIter([]const u8) {
                // Shrink at codepoint level to preserve UTF-8 validity.
                // Decode to codepoints, try shorter prefixes, then re-encode.
                const State = struct {
                    codepoints: []u21,
                    original_bytes: []const u8,
                    alloc: std.mem.Allocator,
                    next_len: usize, // shorter prefix phase
                    done: bool,

                    fn nextImpl(self: *@This()) ?[]const u8 {
                        if (self.done) return null;

                        // Try shorter codepoint prefixes
                        if (self.next_len < self.codepoints.len) {
                            const len = self.next_len;
                            self.next_len += 1;
                            return self.encode(self.codepoints[0..len]);
                        }
                        self.done = true;
                        return null;
                    }

                    fn encode(self: *@This(), cps: []const u21) ?[]const u8 {
                        // Worst case 4 bytes per codepoint
                        var buf = self.alloc.alloc(u8, cps.len * 4) catch return null;
                        var pos: usize = 0;
                        for (cps) |cp| {
                            const cp_len = std.unicode.utf8CodepointSequenceLength(cp) catch continue;
                            if (pos + cp_len > buf.len) break;
                            _ = std.unicode.utf8Encode(cp, buf[pos..]) catch continue;
                            pos += cp_len;
                        }
                        return buf[0..pos];
                    }

                    fn typeErasedNext(ctx: *anyopaque) ?[]const u8 {
                        const self: *@This() = @ptrCast(@alignCast(ctx));
                        return self.nextImpl();
                    }
                };

                // Decode UTF-8 to codepoints
                const view = std.unicode.Utf8View.init(value) catch return ShrinkIter([]const u8).empty();
                var cp_iter = view.iterator();
                var count: usize = 0;
                while (cp_iter.nextCodepoint()) |_| count += 1;

                if (count == 0) return ShrinkIter([]const u8).empty();

                const cps = allocator.alloc(u21, count) catch return ShrinkIter([]const u8).empty();
                var cp_iter2 = view.iterator();
                var i: usize = 0;
                while (cp_iter2.nextCodepoint()) |cp| : (i += 1) {
                    cps[i] = cp;
                }

                const state = allocator.create(State) catch return ShrinkIter([]const u8).empty();
                state.* = .{
                    .codepoints = cps,
                    .original_bytes = value,
                    .alloc = allocator,
                    .next_len = 0,
                    .done = false,
                };
                return .{
                    .context = @ptrCast(state),
                    .nextFn = State.typeErasedNext,
                };
            }
        }.f,
    };
}

// -- Collection generators ------------------------------------------------

/// Generate a random permutation of a comptime-known slice.
/// Shrinks toward the original (sorted) order by trying to undo swaps.
pub fn shuffle(comptime T: type, comptime items: []const T) Gen([]const T) {
    return .{
        .genFn = struct {
            fn gen(rng: std.Random, allocator: std.mem.Allocator, _: usize) []const T {
                const buf = allocator.alloc(T, items.len) catch return items;
                @memcpy(buf, items);
                rng.shuffle(T, buf);
                return buf;
            }
        }.gen,
        .shrinkFn = struct {
            fn f(value: []const T, allocator: std.mem.Allocator) ShrinkIter([]const T) {
                // Shrink toward original order by undoing adjacent out-of-order swaps.
                if (value.len < 2) return ShrinkIter([]const T).empty();

                const State = struct {
                    original: []const T,
                    alloc: std.mem.Allocator,
                    pos: usize,

                    fn nextImpl(ctx: *anyopaque) ?[]const T {
                        const self: *@This() = @ptrCast(@alignCast(ctx));
                        while (self.pos + 1 < self.original.len) {
                            const i = self.pos;
                            self.pos += 1;
                            // Check if swapping would move toward sorted order
                            const idx_a = originalIndex(self.original[i]);
                            const idx_b = originalIndex(self.original[i + 1]);
                            if (idx_a > idx_b) {
                                // Out of order relative to items -- yield swapped copy
                                const buf = self.alloc.alloc(T, self.original.len) catch return null;
                                @memcpy(buf, self.original);
                                buf[i] = self.original[i + 1];
                                buf[i + 1] = self.original[i];
                                return buf;
                            }
                        }
                        return null;
                    }

                    fn originalIndex(val: T) usize {
                        for (items, 0..) |item, i| {
                            if (std.meta.eql(item, val)) return i;
                        }
                        return items.len; // not found
                    }
                };
                const state = allocator.create(State) catch return ShrinkIter([]const T).empty();
                state.* = .{ .original = value, .alloc = allocator, .pos = 0 };
                return .{
                    .context = @ptrCast(state),
                    .nextFn = State.nextImpl,
                };
            }
        }.f,
    };
}

/// Generate a random subsequence (sublist) of a comptime-known slice.
/// Each element is independently included or excluded.
pub fn sublistOf(comptime T: type, comptime items: []const T) Gen([]const T) {
    return .{
        .genFn = struct {
            fn gen(rng: std.Random, allocator: std.mem.Allocator, _: usize) []const T {
                // Count how many we'll include
                var count: usize = 0;
                var include: [items.len]bool = undefined;
                for (0..items.len) |i| {
                    include[i] = rng.boolean();
                    if (include[i]) count += 1;
                }
                const buf = allocator.alloc(T, count) catch return &.{};
                var idx: usize = 0;
                for (0..items.len) |i| {
                    if (include[i]) {
                        buf[idx] = items[i];
                        idx += 1;
                    }
                }
                return buf;
            }
        }.gen,
        .shrinkFn = struct {
            fn f(value: []const T, allocator: std.mem.Allocator) ShrinkIter([]const T) {
                // Shrink by trying to remove each element
                return sliceRange(T, int_or_element(T), 0, items.len).shrinkFn(value, allocator);
            }

            fn int_or_element(comptime U: type) Gen(U) {
                // Dummy generator -- only the shrinker is used by sliceRange.
                return .{
                    .genFn = struct {
                        fn f(_: std.Random, _: std.mem.Allocator, _: usize) U {
                            return std.mem.zeroes(U);
                        }
                    }.f,
                    .shrinkFn = struct {
                        fn f(_: U, _: std.mem.Allocator) ShrinkIter(U) {
                            return ShrinkIter(U).empty();
                        }
                    }.f,
                };
            }
        }.f,
    };
}

/// Generate a sorted slice of T. Generates a random slice, then sorts it.
/// Requires T to be comparable via `<`.
pub fn orderedList(comptime T: type, comptime inner: Gen(T), comptime max_len: usize) Gen([]const T) {
    return .{
        .genFn = struct {
            fn gen(rng: std.Random, allocator: std.mem.Allocator, size: usize) []const T {
                const len = rng.intRangeAtMost(usize, 0, max_len);
                const buf = allocator.alloc(T, len) catch return &.{};
                for (buf) |*slot| {
                    slot.* = inner.generate(rng, allocator, size);
                }
                std.mem.sort(T, buf, {}, struct {
                    fn lessThan(_: void, a: T, b: T) bool {
                        return a < b;
                    }
                }.lessThan);
                return buf;
            }
        }.gen,
        .shrinkFn = struct {
            fn f(value: []const T, allocator: std.mem.Allocator) ShrinkIter([]const T) {
                // Wrap the slice shrinker and re-sort each candidate to maintain
                // the sorted invariant.
                const State = struct {
                    inner_iter: ShrinkIter([]const T),
                    alloc: std.mem.Allocator,

                    fn nextSorted(self: *@This()) ?[]const T {
                        while (self.inner_iter.next()) |candidate| {
                            // Make a mutable copy to sort
                            const buf = self.alloc.alloc(T, candidate.len) catch return null;
                            @memcpy(buf, candidate);
                            std.mem.sort(T, buf, {}, struct {
                                fn lessThan(_: void, a: T, b: T) bool {
                                    return a < b;
                                }
                            }.lessThan);
                            return buf;
                        }
                        return null;
                    }

                    fn typeErasedNext(ctx: *anyopaque) ?[]const T {
                        const self: *@This() = @ptrCast(@alignCast(ctx));
                        return self.nextSorted();
                    }
                };

                const inner_iter = slice(T, inner, max_len).shrinkFn(value, allocator);
                const state = allocator.create(State) catch return ShrinkIter([]const T).empty();
                state.* = .{
                    .inner_iter = inner_iter,
                    .alloc = allocator,
                };
                return .{
                    .context = @ptrCast(state),
                    .nextFn = State.typeErasedNext,
                };
            }
        }.f,
    };
}

/// Generate values from a comptime-known list, biasing toward earlier elements.
/// Uses min-of-three random indices to create a distribution that strongly
/// favors the beginning of the list.
pub fn growingElements(comptime T: type, comptime items: []const T) Gen(T) {
    comptime {
        if (items.len == 0) @compileError("growingElements() requires at least one item");
    }
    return .{
        .genFn = struct {
            fn gen(rng: std.Random, _: std.mem.Allocator, _: usize) T {
                // Min-of-three gives a steeper bias toward index 0.
                const idx1 = rng.intRangeAtMost(usize, 0, items.len - 1);
                const idx2 = rng.intRangeAtMost(usize, 0, items.len - 1);
                const idx3 = rng.intRangeAtMost(usize, 0, items.len - 1);
                return items[@min(idx1, @min(idx2, idx3))];
            }
        }.gen,
        .shrinkFn = struct {
            fn f(value: T, allocator: std.mem.Allocator) ShrinkIter(T) {
                return combinators.element(T, items).shrinkFn(value, allocator);
            }
        }.f,
    };
}

/// Sample N values from a generator. Useful for debugging generators.
/// Returns a heap-allocated slice of generated values.
/// Logs the seed used so results can be reproduced with `sampleWith`.
pub fn sample(comptime T: type, gen: Gen(T), n: usize, allocator: std.mem.Allocator) ![]T {
    const seed = @as(u64, @truncate(@as(u128, @bitCast(std.time.nanoTimestamp()))));
    std.log.info("zigcheck.sample: using seed 0x{x} (reproduce with sampleWith(..., 0x{x}, ...))", .{ seed, seed });
    return sampleWith(T, gen, n, seed, allocator);
}

/// Sample N values with a specific seed for reproducibility.
pub fn sampleWith(comptime T: type, gen: Gen(T), n: usize, seed: u64, allocator: std.mem.Allocator) ![]T {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const result = try allocator.alloc(T, n);
    for (result) |*slot| {
        slot.* = gen.generate(rng, allocator, 100);
    }
    return result;
}

// -- Tests ----------------------------------------------------------------

test "asciiChar: produces printable ASCII" {
    var prng = std.Random.DefaultPrng.init(42);
    const g = asciiChar();
    for (0..200) |_| {
        const v = g.generate(prng.random(), std.testing.allocator, 100);
        try std.testing.expect(v >= 32 and v <= 126);
    }
}

test "alphanumeric: produces only alphanumeric chars" {
    var prng = std.Random.DefaultPrng.init(42);
    const g = alphanumeric();
    for (0..200) |_| {
        const v = g.generate(prng.random(), std.testing.allocator, 100);
        const is_alnum = (v >= 'a' and v <= 'z') or (v >= 'A' and v <= 'Z') or (v >= '0' and v <= '9');
        try std.testing.expect(is_alnum);
    }
}

test "slice: generates slices within length bounds" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    var prng = std.Random.DefaultPrng.init(42);
    const g = slice(u8, generators.int(u8), 10);
    for (0..100) |_| {
        const v = g.generate(prng.random(), arena_state.allocator(), 100);
        try std.testing.expect(v.len <= 10);
    }
}

test "sliceRange: respects min and max length" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    var prng = std.Random.DefaultPrng.init(42);
    const g = sliceRange(u8, generators.int(u8), 3, 8);
    for (0..100) |_| {
        const v = g.generate(prng.random(), arena_state.allocator(), 100);
        try std.testing.expect(v.len >= 3 and v.len <= 8);
    }
}

test "asciiString: generates valid ASCII strings" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    var prng = std.Random.DefaultPrng.init(42);
    const g = asciiString(20);
    for (0..100) |_| {
        const v = g.generate(prng.random(), arena_state.allocator(), 100);
        try std.testing.expect(v.len <= 20);
        for (v) |c| {
            try std.testing.expect(c >= 32 and c <= 126);
        }
    }
}

test "slice shrink: chunk-based removal starts with half" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = slice(u8, generators.int(u8), 10);
    const original = &[_]u8{ 1, 2, 3, 4 };
    var si = g.shrink(original, arena_state.allocator());
    // chunk_size starts at (4-0)/2 = 2; first candidate removes [0..2)
    const first = si.next().?;
    try std.testing.expectEqual(@as(usize, 2), first.len);
    try std.testing.expectEqual(@as(u8, 3), first[0]);
    try std.testing.expectEqual(@as(u8, 4), first[1]);
}

test "slice shrink: all candidates are shorter or have shrunk elements" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = slice(u8, generators.int(u8), 10);
    const original = &[_]u8{ 100, 200 };
    var si = g.shrink(original, arena_state.allocator());

    // chunk_size = (2-0)/2 = 1, so single-element removals first, then element shrinks
    var count: usize = 0;
    while (si.next()) |candidate| {
        // All candidates should either be shorter or same length with smaller elements
        try std.testing.expect(candidate.len <= original.len);
        count += 1;
        if (count > 50) break;
    }
    try std.testing.expect(count > 0);
}

test "slice shrink: respects min_len" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = sliceRange(u8, generators.int(u8), 2, 10);
    const original = &[_]u8{ 5, 10, 15 };
    var si = g.shrink(original, arena_state.allocator());
    // All chunk removals must respect min_len=2
    while (si.next()) |candidate| {
        try std.testing.expect(candidate.len >= 2);
    }
}

test "slice shrink: empty slice has no shrinks" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = slice(u8, generators.int(u8), 10);
    const original: []const u8 = &.{};
    var si = g.shrink(original, arena_state.allocator());
    try std.testing.expectEqual(null, si.next());
}

test "shuffle: produces all elements" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    var prng = std.Random.DefaultPrng.init(42);
    const items = [_]u8{ 1, 2, 3, 4 };
    const g = shuffle(u8, &items);
    const result = g.generate(prng.random(), arena_state.allocator(), 100);
    try std.testing.expectEqual(@as(usize, 4), result.len);
    // All original elements should be present
    var sum: u32 = 0;
    for (result) |v| sum += v;
    try std.testing.expectEqual(@as(u32, 10), sum);
}

test "sublistOf: produces subset" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    var prng = std.Random.DefaultPrng.init(42);
    const items = [_]u8{ 10, 20, 30, 40, 50 };
    const g = sublistOf(u8, &items);
    for (0..100) |_| {
        const result = g.generate(prng.random(), arena_state.allocator(), 100);
        try std.testing.expect(result.len <= 5);
        // All elements must come from original
        for (result) |v| {
            var found = false;
            for (items) |item| {
                if (v == item) found = true;
            }
            try std.testing.expect(found);
        }
    }
}

test "orderedList: produces sorted output" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    var prng = std.Random.DefaultPrng.init(42);
    const g = orderedList(u32, generators.int(u32), 20);
    for (0..50) |_| {
        const result = g.generate(prng.random(), arena_state.allocator(), 100);
        try std.testing.expect(result.len <= 20);
        // Verify sorted
        if (result.len > 1) {
            for (0..result.len - 1) |i| {
                try std.testing.expect(result[i] <= result[i + 1]);
            }
        }
    }
}

test "orderedList: shrink candidates are sorted" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = orderedList(u32, generators.int(u32), 10);
    const input = &[_]u32{ 3, 7, 15, 42 };
    var si = g.shrink(input, arena_state.allocator());
    var count: usize = 0;
    while (si.next()) |candidate| {
        // Every shrink candidate must be sorted
        if (candidate.len > 1) {
            for (0..candidate.len - 1) |i| {
                try std.testing.expect(candidate[i] <= candidate[i + 1]);
            }
        }
        count += 1;
        if (count > 50) break; // don't exhaust all candidates
    }
}

test "growingElements: min-of-three biases toward early elements" {
    var prng = std.Random.DefaultPrng.init(42);
    const items = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const g = growingElements(u32, &items);
    var sum: u64 = 0;
    const n = 1000;
    for (0..n) |_| {
        sum += g.generate(prng.random(), std.testing.allocator, 100);
    }
    // With min-of-three bias, average should be well below 3.0 (uniform midpoint is 4.5)
    const avg = @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(n));
    try std.testing.expect(avg < 3.0);
}

test "sample: returns N values" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const result = try sampleWith(u32, generators.int(u32), 10, 42, arena_state.allocator());
    try std.testing.expectEqual(@as(usize, 10), result.len);
}

test "unicodeChar: produces valid code points" {
    var prng = std.Random.DefaultPrng.init(42);
    const g = unicodeChar();
    for (0..200) |_| {
        const cp = g.generate(prng.random(), std.testing.allocator, 100);
        try std.testing.expect(cp <= 0x10FFFF);
        // Not a surrogate
        try std.testing.expect(cp < 0xD800 or cp > 0xDFFF);
    }
}

test "unicodeChar: shrinker never emits surrogates" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = unicodeChar();
    // Shrink a high code point -- binary search will pass through surrogate range
    var si = g.shrink(0x10000, arena_state.allocator());
    while (si.next()) |cp| {
        try std.testing.expect(cp < 0xD800 or cp > 0xDFFF);
        try std.testing.expect(cp <= 0x10FFFF);
    }
}

test "unicodeString: produces valid UTF-8" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    var prng = std.Random.DefaultPrng.init(42);
    const g = unicodeString(20);
    for (0..50) |_| {
        const s = g.generate(prng.random(), arena_state.allocator(), 100);
        // Validate UTF-8 by attempting to create a view
        const valid = if (std.unicode.Utf8View.init(s)) |_| true else |_| false;
        try std.testing.expect(valid);
    }
}

test "shuffle: shrinker tries to restore original order" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const items = &[_]u32{ 1, 2, 3 };
    const g = shuffle(u32, items);
    // Reversed order should produce swap candidates
    const reversed = &[_]u32{ 3, 2, 1 };
    var si = g.shrink(reversed, arena_state.allocator());
    const first = si.next().?;
    // First swap should be at position 0: swap 3,2 -> [2,3,1]
    try std.testing.expectEqual(@as(u32, 2), first[0]);
    try std.testing.expectEqual(@as(u32, 3), first[1]);
    try std.testing.expectEqual(@as(u32, 1), first[2]);
}

test "sliceOfRange + element: custom charset string generation" {
    // Verify the pattern: sliceOfRange(element(u8, charset), min, max) works
    // for generating strings from arbitrary character classes.
    const charset = "abcdefghijklmnopqrstuvwxyz0123456789";
    const g = comptime combinators.sliceOfRange(combinators.element(u8, charset), 1, 12);

    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    var prng = std.Random.DefaultPrng.init(42);

    for (0..100) |_| {
        const val = g.generate(prng.random(), arena_state.allocator(), 100);
        try std.testing.expect(val.len >= 1 and val.len <= 12);
        for (val) |c| {
            var found = false;
            for (charset) |valid| {
                if (c == valid) {
                    found = true;
                    break;
                }
            }
            try std.testing.expect(found);
        }
    }
}

test "shuffle: already sorted has no shrinks" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const items = &[_]u32{ 1, 2, 3 };
    const g = shuffle(u32, items);
    var si = g.shrink(items, arena_state.allocator());
    try std.testing.expectEqual(null, si.next());
}
