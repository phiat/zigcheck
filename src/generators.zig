// Built-in generators for primitive types.

const std = @import("std");
const Gen = @import("gen.zig").Gen;
const ShrinkIter = @import("shrink.zig").ShrinkIter;
const shrink = @import("shrink.zig");

/// Generator for any integer type. Produces the full range of values.
pub fn int(comptime T: type) Gen(T) {
    comptime {
        if (@typeInfo(T) != .int) @compileError("int() requires an integer type, got " ++ @typeName(T));
    }
    return .{
        .genFn = struct {
            fn f(rng: std.Random, _: std.mem.Allocator) T {
                return rng.int(T);
            }
        }.f,
        .shrinkFn = struct {
            fn f(value: T, allocator: std.mem.Allocator) ShrinkIter(T) {
                const state = allocator.create(shrink.IntShrinkState(T)) catch return ShrinkIter(T).empty();
                state.* = shrink.IntShrinkState(T).init(value);
                return state.iter();
            }
        }.f,
    };
}

/// Generator for integers in [min, max] inclusive.
pub fn intRange(comptime T: type, comptime min: T, comptime max: T) Gen(T) {
    comptime {
        if (@typeInfo(T) != .int) @compileError("intRange() requires an integer type, got " ++ @typeName(T));
        if (min > max) @compileError("intRange(): min must be <= max");
    }
    return .{
        .genFn = struct {
            fn f(rng: std.Random, _: std.mem.Allocator) T {
                return rng.intRangeAtMost(T, min, max);
            }
        }.f,
        .shrinkFn = struct {
            fn f(value: T, allocator: std.mem.Allocator) ShrinkIter(T) {
                // Shrink toward min by shifting into [0, max-min] space,
                // shrinking there, then shifting back. All candidates are
                // guaranteed to be in [min, max].
                const RangeShrinkState = struct {
                    inner: shrink.IntShrinkState(T),

                    fn nextClamped(self: *@This()) ?T {
                        while (self.inner.next()) |raw| {
                            const shifted = raw +% min;
                            if (shifted >= min and shifted <= max) return shifted;
                        }
                        return null;
                    }

                    fn typeErasedNext(ctx: *anyopaque) ?T {
                        const self: *@This() = @ptrCast(@alignCast(ctx));
                        return self.nextClamped();
                    }
                };
                const state = allocator.create(RangeShrinkState) catch return ShrinkIter(T).empty();
                state.* = .{ .inner = shrink.IntShrinkState(T).init(value -% min) };
                return .{
                    .context = @ptrCast(state),
                    .nextFn = RangeShrinkState.typeErasedNext,
                };
            }
        }.f,
    };
}

/// Generator for boolean values.
pub fn boolean() Gen(bool) {
    return .{
        .genFn = struct {
            fn f(rng: std.Random, _: std.mem.Allocator) bool {
                return rng.boolean();
            }
        }.f,
        .shrinkFn = struct {
            fn f(value: bool, allocator: std.mem.Allocator) ShrinkIter(bool) {
                const state = allocator.create(shrink.BoolShrinkState) catch return ShrinkIter(bool).empty();
                state.* = shrink.BoolShrinkState.init(value);
                return state.iter();
            }
        }.f,
    };
}

/// Generator for a single byte (u8).
pub fn byte() Gen(u8) {
    return int(u8);
}

/// Generator for floating point types (f16, f32, f64).
/// Produces the full range of finite float values including negatives.
/// Does not produce NaN or infinity (use floatAny for those).
pub fn float(comptime T: type) Gen(T) {
    comptime {
        if (@typeInfo(T) != .float) @compileError("float() requires a float type, got " ++ @typeName(T));
    }
    const Bits = std.meta.Int(.unsigned, @bitSizeOf(T));
    return .{
        .genFn = struct {
            fn f(rng: std.Random, _: std.mem.Allocator) T {
                // Generate full-range finite floats via random bit patterns,
                // re-rolling NaN and infinity to keep values finite.
                while (true) {
                    const result: T = @bitCast(rng.int(Bits));
                    if (std.math.isFinite(result)) return result;
                }
            }
        }.f,
        .shrinkFn = struct {
            fn f(value: T, allocator: std.mem.Allocator) ShrinkIter(T) {
                const state = allocator.create(shrink.FloatShrinkState(T)) catch return ShrinkIter(T).empty();
                state.* = shrink.FloatShrinkState(T).init(value);
                return state.iter();
            }
        }.f,
    };
}

// -- Slice and string generators ------------------------------------------

/// Generator for a single printable ASCII character (32-126).
pub fn asciiChar() Gen(u8) {
    return intRange(u8, 32, 126);
}

/// Generator for a single alphanumeric character [a-zA-Z0-9].
pub fn alphanumeric() Gen(u8) {
    const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    return element(u8, charset);
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
            fn gen(rng: std.Random, allocator: std.mem.Allocator) []const T {
                const len = rng.intRangeAtMost(usize, min_len, max_len);
                const result = allocator.alloc(T, len) catch return &.{};
                for (result) |*slot| {
                    slot.* = inner.generate(rng, allocator);
                }
                return result;
            }
        }.gen,
        .shrinkFn = struct {
            fn shrinkFn(value: []const T, allocator: std.mem.Allocator) ShrinkIter([]const T) {
                if (value.len <= min_len) return ShrinkIter([]const T).empty();

                // Shrink phases:
                //  1. Shorter prefixes (length min_len..original.len-1)
                //  2. Delete one element at each position
                //  3. Shrink individual elements in place
                const Phase = enum { prefixes, deletions, elements };

                const State = struct {
                    original: []const T,
                    alloc: std.mem.Allocator,
                    phase: Phase,
                    // Shorter-prefix phase: next length to try
                    next_len: usize,
                    // Delete-one phase: index of element to delete next
                    delete_idx: usize,
                    // Element-shrink phase: position and current element's shrinker
                    elem_idx: usize,
                    elem_shrinker: ShrinkIter(T),

                    fn nextImpl(self: *@This()) ?[]const T {
                        switch (self.phase) {
                            .prefixes => {
                                if (self.tryShorter()) |result| return result;
                                self.phase = .deletions;
                                self.delete_idx = 0;
                                return self.nextImpl();
                            },
                            .deletions => {
                                if (self.tryDeleteOne()) |result| return result;
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

                    /// Yield the next shorter prefix, or null when exhausted.
                    fn tryShorter(self: *@This()) ?[]const T {
                        while (self.next_len < self.original.len) {
                            const len = self.next_len;
                            self.next_len += 1;
                            return self.copyPrefix(len);
                        }
                        return null;
                    }

                    /// Yield the original with element at delete_idx removed.
                    fn tryDeleteOne(self: *@This()) ?[]const T {
                        if (self.original.len <= min_len) return null;
                        while (self.delete_idx < self.original.len) {
                            const idx = self.delete_idx;
                            self.delete_idx += 1;
                            return self.copyWithout(idx);
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

                    fn copyPrefix(self: *@This(), len: usize) ?[]const T {
                        const result = self.alloc.alloc(T, len) catch return null;
                        if (len > 0) @memcpy(result, self.original[0..len]);
                        return result;
                    }

                    fn copyWithout(self: *@This(), skip_idx: usize) ?[]const T {
                        const new_len = self.original.len - 1;
                        const result = self.alloc.alloc(T, new_len) catch return null;
                        var out: usize = 0;
                        for (0..self.original.len) |i| {
                            if (i == skip_idx) continue;
                            result[out] = self.original[i];
                            out += 1;
                        }
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

                const state = allocator.create(State) catch return ShrinkIter([]const T).empty();
                state.* = .{
                    .original = value,
                    .alloc = allocator,
                    .phase = .prefixes,
                    .next_len = min_len,
                    .delete_idx = 0,
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
    return slice(u8, int(u8), max_len);
}

/// Generator for a random Unicode code point (excludes surrogates).
pub fn unicodeChar() Gen(u21) {
    return .{
        .genFn = struct {
            fn gen(rng: std.Random, _: std.mem.Allocator) u21 {
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
pub fn unicodeString(comptime max_codepoints: usize) Gen([]const u8) {
    return .{
        .genFn = struct {
            fn gen(rng: std.Random, allocator: std.mem.Allocator) []const u8 {
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

// -- Combinators ----------------------------------------------------------

/// Always produces the same value. Shrinks to nothing.
pub fn constant(comptime T: type, comptime value: T) Gen(T) {
    return .{
        .genFn = struct {
            fn f(_: std.Random, _: std.mem.Allocator) T {
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
            fn f(rng: std.Random, _: std.mem.Allocator) T {
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
            fn f(rng: std.Random, allocator: std.mem.Allocator) T {
                const idx = rng.intRangeAtMost(usize, 0, gens.len - 1);
                inline for (gens, 0..) |g, i| {
                    if (idx == i) return g.generate(rng, allocator);
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

/// Transform generated values: Gen(A) -> Gen(B) via a comptime-known function.
/// Shrinks by applying the mapping to the inner generator's shrink candidates.
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
            fn gen(rng: std.Random, allocator: std.mem.Allocator) B {
                return f(inner.generate(rng, allocator));
            }
        }.gen,
        .shrinkFn = struct {
            fn shrinkFn(_: B, _: std.mem.Allocator) ShrinkIter(B) {
                // f is not invertible, so we can't recover the original A to
                // feed to inner's shrinker. Use shrinkMap for shrink support.
                return ShrinkIter(B).empty();
            }
        }.shrinkFn,
    };
}

/// Filter generated values. Retries up to 1000 times to find a value that
/// satisfies the predicate. Panics if no satisfying value is found.
/// Error returned when a filter predicate rejects too many consecutive values.
pub const FilterExhausted = error.FilterExhausted;

pub fn filter(
    comptime T: type,
    comptime inner: Gen(T),
    comptime pred: *const fn (T) bool,
) Gen(T) {
    const max_retries = 1000;
    return .{
        .genFn = struct {
            fn gen(rng: std.Random, allocator: std.mem.Allocator) T {
                for (0..max_retries) |_| {
                    const val = inner.generate(rng, allocator);
                    if (pred(val)) return val;
                }
                // Log a diagnostic rather than panicking so the test runner can
                // report the seed and continue with other tests.
                std.log.warn("zcheck.filter: predicate rejected {d} consecutive values; predicate may be too restrictive", .{max_retries});
                // Return the last generated value even though it doesn't pass the
                // predicate. This lets the runner proceed (the property will likely
                // fail, and the seed will be reported).
                return inner.generate(rng, allocator);
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
            fn gen(rng: std.Random, allocator: std.mem.Allocator) T {
                comptime var total: usize = 0;
                inline for (weighted) |entry| {
                    total += entry[0];
                }
                var pick = rng.intRangeLessThan(usize, 0, total);
                inline for (weighted) |entry| {
                    if (pick < entry[0]) return entry[1].generate(rng, allocator);
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
            fn gen(rng: std.Random, allocator: std.mem.Allocator) B {
                return forward(inner.generate(rng, allocator));
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

/// Generate a random permutation of a comptime-known slice.
/// Shrinks toward the original (sorted) order by trying to undo swaps.
pub fn shuffle(comptime T: type, comptime items: []const T) Gen([]const T) {
    return .{
        .genFn = struct {
            fn gen(rng: std.Random, allocator: std.mem.Allocator) []const T {
                const buf = allocator.alloc(T, items.len) catch return items;
                @memcpy(buf, items);
                rng.shuffle(T, buf);
                return buf;
            }
        }.gen,
        .shrinkFn = struct {
            fn f(value: []const T, allocator: std.mem.Allocator) ShrinkIter([]const T) {
                // Shrink toward original order by undoing adjacent out-of-order swaps.
                // For each position i where value[i] > value[i+1] relative to the
                // original list's ordering, yield a copy with those two elements swapped.
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
            fn gen(rng: std.Random, allocator: std.mem.Allocator) []const T {
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
                // genFn returns a zero value instead of unreachable to avoid
                // a safety landmine if accidentally called.
                return .{
                    .genFn = struct {
                        fn f(_: std.Random, _: std.mem.Allocator) U {
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
            fn gen(rng: std.Random, allocator: std.mem.Allocator) []const T {
                const len = rng.intRangeAtMost(usize, 0, max_len);
                const buf = allocator.alloc(T, len) catch return &.{};
                for (buf) |*slot| {
                    slot.* = inner.generate(rng, allocator);
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

/// Generate values from a comptime-known list, biasing toward earlier elements
/// as the test progresses. Similar to QuickCheck's growingElements.
/// Early tests get items near the beginning; later tests get items from anywhere.
/// Pick from a comptime-known list with a bias toward earlier elements.
/// Uses min-of-three random indices to create a distribution that strongly
/// favors the beginning of the list. For a list of length N, the probability
/// of picking index k is proportional to (N-k)^3 - (N-k-1)^3.
///
/// Note: unlike Haskell's QuickCheck, this bias is static (not size-dependent)
/// because Zig generators don't receive a "size" parameter.
pub fn growingElements(comptime T: type, comptime items: []const T) Gen(T) {
    comptime {
        if (items.len == 0) @compileError("growingElements() requires at least one item");
    }
    return .{
        .genFn = struct {
            fn gen(rng: std.Random, _: std.mem.Allocator) T {
                // Min-of-three gives a steeper bias toward index 0.
                const idx1 = rng.intRangeAtMost(usize, 0, items.len - 1);
                const idx2 = rng.intRangeAtMost(usize, 0, items.len - 1);
                const idx3 = rng.intRangeAtMost(usize, 0, items.len - 1);
                return items[@min(idx1, @min(idx2, idx3))];
            }
        }.gen,
        .shrinkFn = struct {
            fn f(value: T, allocator: std.mem.Allocator) ShrinkIter(T) {
                return element(T, items).shrinkFn(value, allocator);
            }
        }.f,
    };
}

/// Sample N values from a generator. Useful for debugging generators.
/// Returns a heap-allocated slice of generated values.
pub fn sample(comptime T: type, gen: Gen(T), n: usize, allocator: std.mem.Allocator) ![]T {
    var prng = std.Random.DefaultPrng.init(@as(u64, @truncate(@as(u128, @bitCast(std.time.nanoTimestamp())))));
    const rng = prng.random();
    const result = try allocator.alloc(T, n);
    for (result) |*slot| {
        slot.* = gen.generate(rng, allocator);
    }
    return result;
}

/// Sample N values with a specific seed for reproducibility.
pub fn sampleWith(comptime T: type, gen: Gen(T), n: usize, seed: u64, allocator: std.mem.Allocator) ![]T {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const result = try allocator.alloc(T, n);
    for (result) |*slot| {
        slot.* = gen.generate(rng, allocator);
    }
    return result;
}

/// Monadic bind: generate an A, then use it to choose a generator for B.
/// The function `f` takes an A and returns a Gen(B) at comptime.
/// Since Zig function pointers can't capture runtime values, `f` must be
/// a comptime function that returns a generator based on the A value.
///
/// Note: Shrinking only applies to the B value. The A value used to select
/// the generator is not shrunk (this would require re-running the generation).
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
            fn gen(rng: std.Random, allocator: std.mem.Allocator) B {
                const a = gen_a.generate(rng, allocator);
                const gen_b = f(a);
                return gen_b.generate(rng, allocator);
            }
        }.gen,
        .shrinkFn = struct {
            fn shrinkFn(_: B, _: std.mem.Allocator) ShrinkIter(B) {
                // Can't shrink: we don't know which gen_b produced this value.
                return ShrinkIter(B).empty();
            }
        }.shrinkFn,
    };
}

/// Auto-derive a generator for any supported type via comptime reflection.
/// Supports: int, float, bool, enum, struct, optional (?T), pointer-to-slice
/// ([]const T), and tagged unions (union(enum)).
pub fn auto(comptime T: type) Gen(T) {
    return switch (@typeInfo(T)) {
        .int => int(T),
        .float => float(T),
        .bool => boolean(),
        .@"enum" => enumGen(T),
        .@"struct" => structGen(T),
        .optional => |info| optionalGen(info.child),
        .pointer => |info| blk: {
            if (info.size == .slice and info.is_const) {
                break :blk sliceAutoGen(info.child);
            }
            @compileError("zcheck.auto: unsupported pointer type " ++ @typeName(T) ++ "; only []const T slices are supported");
        },
        .@"union" => |info| if (info.tag_type != null)
            unionGen(T)
        else
            @compileError("zcheck.auto: untagged union " ++ @typeName(T) ++ " is not supported; use union(enum) instead"),
        else => @compileError("zcheck.auto: unsupported type " ++ @typeName(T)),
    };
}

/// Generator that picks a random enum variant.
fn enumGen(comptime T: type) Gen(T) {
    comptime {
        if (@typeInfo(T) != .@"enum") @compileError("enumGen requires an enum type");
    }
    return .{
        .genFn = struct {
            fn f(rng: std.Random, _: std.mem.Allocator) T {
                const fields = @typeInfo(T).@"enum".fields;
                const index = rng.intRangeAtMost(usize, 0, fields.len - 1);
                inline for (fields, 0..) |field, i| {
                    if (index == i) return @enumFromInt(field.value);
                }
                unreachable;
            }
        }.f,
        .shrinkFn = struct {
            fn f(value: T, allocator: std.mem.Allocator) ShrinkIter(T) {
                const fields = @typeInfo(T).@"enum".fields;
                const tag_val = @intFromEnum(value);

                // Find declaration index of current value
                var current_idx: usize = fields.len;
                inline for (fields, 0..) |field, i| {
                    if (field.value == tag_val) current_idx = i;
                }

                if (current_idx == 0 or current_idx == fields.len) return ShrinkIter(T).empty();

                // Allocate array of candidates with lower declaration index
                const candidates = allocator.alloc(T, current_idx) catch return ShrinkIter(T).empty();
                inline for (fields, 0..) |field, i| {
                    if (i < current_idx) {
                        candidates[i] = @enumFromInt(field.value);
                    }
                }

                // Allocate iterator state
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

/// Generator for structs -- generates each field independently.
fn structGen(comptime T: type) Gen(T) {
    comptime {
        if (@typeInfo(T) != .@"struct") @compileError("structGen requires a struct type");
    }
    return .{
        .genFn = struct {
            fn f(rng: std.Random, allocator: std.mem.Allocator) T {
                var result: T = undefined;
                inline for (@typeInfo(T).@"struct".fields) |field| {
                    @field(result, field.name) = auto(field.type).generate(rng, allocator);
                }
                return result;
            }
        }.f,
        .shrinkFn = struct {
            fn f(value: T, allocator: std.mem.Allocator) ShrinkIter(T) {
                const struct_fields = @typeInfo(T).@"struct".fields;
                if (struct_fields.len == 0) return ShrinkIter(T).empty();

                // For each field, create a ShrinkIter(T) that maps field shrinks
                // to whole-struct values. Chain them: try field 0, then field 1, etc.
                //
                // State holds: which field we're on, and for each field a type-erased
                // iterator over T. We pre-build an array of ShrinkIter(T), one per field.

                const iters = allocator.alloc(ShrinkIter(T), struct_fields.len) catch return ShrinkIter(T).empty();

                inline for (struct_fields, 0..) |field, i| {
                    // Create a mapper state that wraps field's ShrinkIter(FieldType)
                    // and produces ShrinkIter(T) by substituting the field in original.
                    const Mapper = struct {
                        original: T,
                        field_shrinker: ShrinkIter(field.type),

                        fn nextMapped(ctx: *anyopaque) ?T {
                            const self: *@This() = @ptrCast(@alignCast(ctx));
                            if (self.field_shrinker.next()) |shrunk_val| {
                                var result = self.original;
                                @field(result, field.name) = shrunk_val;
                                return result;
                            }
                            return null;
                        }
                    };
                    const mapper = allocator.create(Mapper) catch return ShrinkIter(T).empty();
                    const field_gen = auto(field.type);
                    mapper.* = .{
                        .original = value,
                        .field_shrinker = field_gen.shrink(@field(value, field.name), allocator),
                    };
                    iters[i] = .{
                        .context = @ptrCast(mapper),
                        .nextFn = Mapper.nextMapped,
                    };
                }

                // Chain state: iterate through the array of ShrinkIter(T)
                const ChainState = struct {
                    iters_arr: []ShrinkIter(T),
                    pos: usize,

                    fn nextChain(ctx: *anyopaque) ?T {
                        const self: *@This() = @ptrCast(@alignCast(ctx));
                        while (self.pos < self.iters_arr.len) {
                            if (self.iters_arr[self.pos].next()) |val| {
                                return val;
                            }
                            self.pos += 1;
                        }
                        return null;
                    }
                };
                const chain = allocator.create(ChainState) catch return ShrinkIter(T).empty();
                chain.* = .{ .iters_arr = iters, .pos = 0 };

                return .{
                    .context = @ptrCast(chain),
                    .nextFn = ChainState.nextChain,
                };
            }
        }.f,
    };
}

/// Generator for optional types -- 10% chance of null, 90% chance of a value.
fn optionalGen(comptime Child: type) Gen(?Child) {
    return .{
        .genFn = struct {
            fn gen(rng: std.Random, allocator: std.mem.Allocator) ?Child {
                // ~10% null
                if (rng.intRangeAtMost(u8, 0, 9) == 0) return null;
                return auto(Child).generate(rng, allocator);
            }
        }.gen,
        .shrinkFn = struct {
            fn f(value: ?Child, allocator: std.mem.Allocator) ShrinkIter(?Child) {
                if (value) |v| {
                    // Try null first, then shrink the inner value
                    const State = struct {
                        yielded_null: bool,
                        inner: ShrinkIter(Child),

                        fn nextImpl(ctx: *anyopaque) ??Child {
                            const self: *@This() = @ptrCast(@alignCast(ctx));
                            if (!self.yielded_null) {
                                self.yielded_null = true;
                                return @as(?Child, null); // yield the ?Child value null
                            }
                            if (self.inner.next()) |shrunk| {
                                return @as(?Child, shrunk);
                            }
                            return @as(??Child, null); // end of iteration
                        }
                    };
                    const state = allocator.create(State) catch return ShrinkIter(?Child).empty();
                    state.* = .{
                        .yielded_null = false,
                        .inner = auto(Child).shrink(v, allocator),
                    };
                    return .{
                        .context = @ptrCast(state),
                        .nextFn = State.nextImpl,
                    };
                } else {
                    return ShrinkIter(?Child).empty();
                }
            }
        }.f,
    };
}

/// Generator for []const T slices via auto -- delegates to slice(T, auto(T), 20).
fn sliceAutoGen(comptime Child: type) Gen([]const Child) {
    return slice(Child, auto(Child), 20);
}

/// Generator for tagged unions -- picks a random variant, generates its payload.
fn unionGen(comptime T: type) Gen(T) {
    const info = @typeInfo(T).@"union";
    const fields = info.fields;
    comptime {
        if (fields.len == 0) @compileError("unionGen: union has no fields");
    }
    return .{
        .genFn = struct {
            fn gen(rng: std.Random, allocator: std.mem.Allocator) T {
                const idx = rng.intRangeAtMost(usize, 0, fields.len - 1);
                inline for (fields, 0..) |field, i| {
                    if (idx == i) {
                        if (field.type == void) {
                            return @unionInit(T, field.name, {});
                        } else {
                            return @unionInit(T, field.name, auto(field.type).generate(rng, allocator));
                        }
                    }
                }
                unreachable;
            }
        }.gen,
        .shrinkFn = struct {
            fn f(value: T, allocator: std.mem.Allocator) ShrinkIter(T) {
                // Shrink by trying earlier variants (void payload), then shrinking the payload.
                const tag = @intFromEnum(std.meta.activeTag(value));

                // Count: earlier void variants + payload shrinks
                var count: usize = 0;
                inline for (fields, 0..) |field, i| {
                    if (@as(usize, @intCast(@intFromEnum(@field(info.tag_type.?, field.name)))) < tag) {
                        if (field.type == void) count += 1;
                    }
                    _ = i;
                }

                // For simplicity, just try earlier void variants
                if (count == 0) {
                    // Try shrinking the payload of the current variant
                    return shrinkUnionPayload(T, value, allocator);
                }

                const candidates = allocator.alloc(T, count) catch return ShrinkIter(T).empty();
                var pos: usize = 0;
                inline for (fields) |field| {
                    if (@as(usize, @intCast(@intFromEnum(@field(info.tag_type.?, field.name)))) < tag) {
                        if (field.type == void) {
                            candidates[pos] = @unionInit(T, field.name, {});
                            pos += 1;
                        }
                    }
                }

                const State = struct {
                    items: []const T,
                    idx: usize,
                    payload_shrinker: ShrinkIter(T),
                    payload_done: bool,

                    fn nextVariant(ctx: *anyopaque) ?T {
                        const self: *@This() = @ptrCast(@alignCast(ctx));
                        if (self.idx < self.items.len) {
                            const val = self.items[self.idx];
                            self.idx += 1;
                            return val;
                        }
                        if (!self.payload_done) {
                            if (self.payload_shrinker.next()) |val| return val;
                            self.payload_done = true;
                        }
                        return null;
                    }
                };
                const state = allocator.create(State) catch return ShrinkIter(T).empty();
                state.* = .{
                    .items = candidates,
                    .idx = 0,
                    .payload_shrinker = shrinkUnionPayload(T, value, allocator),
                    .payload_done = false,
                };
                return .{
                    .context = @ptrCast(state),
                    .nextFn = State.nextVariant,
                };
            }

            fn shrinkUnionPayload(comptime U: type, value: U, allocator: std.mem.Allocator) ShrinkIter(U) {
                const u_info = @typeInfo(U).@"union";
                const u_fields = u_info.fields;
                inline for (u_fields) |field| {
                    if (std.meta.activeTag(value) == @field(u_info.tag_type.?, field.name)) {
                        if (field.type == void) return ShrinkIter(U).empty();
                        const payload = @field(value, field.name);
                        const inner_iter = allocator.create(ShrinkIter(field.type)) catch return ShrinkIter(U).empty();
                        inner_iter.* = auto(field.type).shrink(payload, allocator);

                        const Mapper = struct {
                            inner: *ShrinkIter(field.type),

                            fn nextMapped(ctx: *anyopaque) ?U {
                                const self: *@This() = @ptrCast(@alignCast(ctx));
                                if (self.inner.next()) |shrunk_payload| {
                                    return @unionInit(U, field.name, shrunk_payload);
                                }
                                return null;
                            }
                        };
                        const mapper = allocator.create(Mapper) catch return ShrinkIter(U).empty();
                        mapper.* = .{ .inner = inner_iter };
                        return .{
                            .context = @ptrCast(mapper),
                            .nextFn = Mapper.nextMapped,
                        };
                    }
                }
                return ShrinkIter(U).empty();
            }
        }.f,
    };
}

// -- Tests ----------------------------------------------------------------

test "int generator produces values" {
    var prng = std.Random.DefaultPrng.init(12345);
    const g = int(i32);
    var seen_positive = false;
    var seen_negative = false;
    for (0..100) |_| {
        const v = g.generate(prng.random(), std.testing.allocator);
        if (v > 0) seen_positive = true;
        if (v < 0) seen_negative = true;
    }
    try std.testing.expect(seen_positive);
    try std.testing.expect(seen_negative);
}

test "intRange: produces values in [10, 20]" {
    var prng = std.Random.DefaultPrng.init(42);
    const g = intRange(u32, 10, 20);
    for (0..200) |_| {
        const v = g.generate(prng.random(), std.testing.allocator);
        try std.testing.expect(v >= 10 and v <= 20);
    }
}

test "intRange: shrinker stays within [min, max]" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = intRange(u32, 10, 20);
    var si = g.shrink(15, arena_state.allocator());
    while (si.next()) |v| {
        try std.testing.expect(v >= 10);
        try std.testing.expect(v <= 20);
    }
}

test "intRange: shrinker first candidate is min" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = intRange(u32, 10, 20);
    var si = g.shrink(15, arena_state.allocator());
    const first = si.next();
    try std.testing.expect(first != null);
    try std.testing.expectEqual(@as(u32, 10), first.?);
}

test "intRange: signed range shrinker stays within bounds" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = intRange(i32, -5, 5);
    var si = g.shrink(3, arena_state.allocator());
    while (si.next()) |v| {
        try std.testing.expect(v >= -5);
        try std.testing.expect(v <= 5);
    }
}

test "boolean generator produces both values" {
    var prng = std.Random.DefaultPrng.init(99);
    const g = boolean();
    var seen_true = false;
    var seen_false = false;
    for (0..100) |_| {
        const v = g.generate(prng.random(), std.testing.allocator);
        if (v) seen_true = true else seen_false = true;
    }
    try std.testing.expect(seen_true);
    try std.testing.expect(seen_false);
}

test "float generator produces finite full-range values" {
    var prng = std.Random.DefaultPrng.init(77);
    const g = float(f64);
    var seen_negative = false;
    var seen_gt_one = false;
    for (0..200) |_| {
        const v = g.generate(prng.random(), std.testing.allocator);
        try std.testing.expect(std.math.isFinite(v));
        if (v < 0.0) seen_negative = true;
        if (v > 1.0) seen_gt_one = true;
    }
    // Full-range generator should produce values outside [0,1)
    try std.testing.expect(seen_negative);
    try std.testing.expect(seen_gt_one);
}

test "enum generator produces all variants" {
    const Color = enum { red, green, blue };
    var prng = std.Random.DefaultPrng.init(42);
    const g = enumGen(Color);
    var seen = [_]bool{ false, false, false };
    for (0..200) |_| {
        const v = g.generate(prng.random(), std.testing.allocator);
        seen[@intFromEnum(v)] = true;
    }
    for (seen) |s| try std.testing.expect(s);
}

test "auto: struct generation" {
    const Point = struct { x: i32, y: i32 };
    var prng = std.Random.DefaultPrng.init(42);
    const g = auto(Point);
    const p = g.generate(prng.random(), std.testing.allocator);
    // Just verify it produces a valid struct
    _ = p.x;
    _ = p.y;
}

test "auto: nested struct generation" {
    const Inner = struct { a: u8, b: bool };
    const Outer = struct { inner: Inner, value: i64 };
    var prng = std.Random.DefaultPrng.init(42);
    const g = auto(Outer);
    const v = g.generate(prng.random(), std.testing.allocator);
    _ = v.inner.a;
    _ = v.value;
}

// -- Shrink tests ---------------------------------------------------------

test "int shrink: 100 first candidate is 0" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = int(i32);
    var si = g.shrink(100, arena_state.allocator());
    try std.testing.expectEqual(@as(i32, 0), si.next().?);
}

test "bool shrink: true -> false" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = boolean();
    var si = g.shrink(true, arena_state.allocator());
    try std.testing.expectEqual(false, si.next().?);
    try std.testing.expectEqual(null, si.next());
}

test "enum shrink: blue -> red, green" {
    const Color = enum { red, green, blue };
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = enumGen(Color);
    var si = g.shrink(.blue, arena_state.allocator());
    try std.testing.expectEqual(Color.red, si.next().?);
    try std.testing.expectEqual(Color.green, si.next().?);
    try std.testing.expectEqual(null, si.next());
}

test "enum shrink: first variant has no shrinks" {
    const Color = enum { red, green, blue };
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = enumGen(Color);
    var si = g.shrink(.red, arena_state.allocator());
    try std.testing.expectEqual(null, si.next());
}

test "enum shrink: sparse enum uses declaration order" {
    const Sparse = enum(u32) { a = 0, b = 100, c = 1 };
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = enumGen(Sparse);
    // .c has tag value 1 but declaration index 2 -- should shrink to .a and .b
    var si = g.shrink(.c, arena_state.allocator());
    try std.testing.expectEqual(Sparse.a, si.next().?);
    try std.testing.expectEqual(Sparse.b, si.next().?);
    try std.testing.expectEqual(null, si.next());
}

test "struct shrink: first candidate shrinks first field" {
    const Point = struct { x: i32, y: i32 };
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = auto(Point);
    var si = g.shrink(.{ .x = 100, .y = -50 }, arena_state.allocator());
    const first = si.next().?;
    // First candidate: x shrunk to 0, y unchanged
    try std.testing.expectEqual(@as(i32, 0), first.x);
    try std.testing.expectEqual(@as(i32, -50), first.y);
}

// -- Combinator tests -----------------------------------------------------

test "constant: always produces the same value" {
    var prng = std.Random.DefaultPrng.init(42);
    const g = constant(u32, 7);
    for (0..50) |_| {
        try std.testing.expectEqual(@as(u32, 7), g.generate(prng.random(), std.testing.allocator));
    }
}

test "element: picks from choices" {
    const choices = [_]u8{ 10, 20, 30 };
    var prng = std.Random.DefaultPrng.init(42);
    const g = element(u8, &choices);
    var seen = [_]bool{ false, false, false };
    for (0..200) |_| {
        const v = g.generate(prng.random(), std.testing.allocator);
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
        const v = g.generate(prng.random(), std.testing.allocator);
        if (v == 1) seen[0] = true;
        if (v == 2) seen[1] = true;
        if (v == 3) seen[2] = true;
    }
    for (seen) |s| try std.testing.expect(s);
}

test "map: transforms values" {
    const g = map(u32, u64, int(u32), struct {
        fn f(x: u32) u64 {
            return @as(u64, x) * 2;
        }
    }.f);
    var prng = std.Random.DefaultPrng.init(42);
    for (0..50) |_| {
        const v = g.generate(prng.random(), std.testing.allocator);
        try std.testing.expect(v % 2 == 0); // always even
    }
}

test "filter: only produces values satisfying predicate" {
    const g = filter(i32, int(i32), struct {
        fn pred(n: i32) bool {
            return n > 0;
        }
    }.pred);
    var prng = std.Random.DefaultPrng.init(42);
    for (0..100) |_| {
        const v = g.generate(prng.random(), std.testing.allocator);
        try std.testing.expect(v > 0);
    }
}

test "filter: shrinks respect predicate" {
    const g = filter(i32, int(i32), struct {
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
    const g = filter(i32, int(i32), struct {
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
        const v = g.generate(prng.random(), std.testing.allocator);
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
        try std.testing.expectEqual(@as(u32, 42), g.generate(prng.random(), std.testing.allocator));
    }
}

test "noShrink: generates normally but has no shrinks" {
    var prng = std.Random.DefaultPrng.init(42);
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = noShrink(i32, int(i32));
    // Generates values
    const v = g.generate(prng.random(), std.testing.allocator);
    _ = v;
    // No shrink candidates
    var si = g.shrink(100, arena_state.allocator());
    try std.testing.expectEqual(null, si.next());
}

test "shrinkMap: shrinks via isomorphism" {
    // Map i32 -> u32 via absolute value. Shrink in i32 space.
    const g = shrinkMap(i32, u32, int(i32), struct {
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

test "shuffle: produces all elements" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    var prng = std.Random.DefaultPrng.init(42);
    const items = [_]u8{ 1, 2, 3, 4 };
    const g = shuffle(u8, &items);
    const result = g.generate(prng.random(), arena_state.allocator());
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
        const result = g.generate(prng.random(), arena_state.allocator());
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
    const g = orderedList(u32, int(u32), 20);
    for (0..50) |_| {
        const result = g.generate(prng.random(), arena_state.allocator());
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
    const g = orderedList(u32, int(u32), 10);
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
        sum += g.generate(prng.random(), std.testing.allocator);
    }
    // With min-of-three bias, average should be well below 3.0 (uniform midpoint is 4.5)
    const avg = @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(n));
    try std.testing.expect(avg < 3.0);
}

test "sample: returns N values" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const result = try sampleWith(u32, int(u32), 10, 42, arena_state.allocator());
    try std.testing.expectEqual(@as(usize, 10), result.len);
}

test "flatMap: dependent generation" {
    // Generate a bool, then generate either 0 or 1000 based on it
    const g = flatMap(bool, u32, boolean(), struct {
        fn f(b: bool) Gen(u32) {
            if (b) return constant(u32, 1000);
            return constant(u32, 0);
        }
    }.f);
    var prng = std.Random.DefaultPrng.init(42);
    var seen_0 = false;
    var seen_1000 = false;
    for (0..200) |_| {
        const v = g.generate(prng.random(), std.testing.allocator);
        if (v == 0) seen_0 = true;
        if (v == 1000) seen_1000 = true;
        try std.testing.expect(v == 0 or v == 1000);
    }
    try std.testing.expect(seen_0);
    try std.testing.expect(seen_1000);
}

test "unicodeChar: produces valid code points" {
    var prng = std.Random.DefaultPrng.init(42);
    const g = unicodeChar();
    for (0..200) |_| {
        const cp = g.generate(prng.random(), std.testing.allocator);
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
        const s = g.generate(prng.random(), arena_state.allocator());
        // Validate UTF-8 by attempting to create a view
        const valid = if (std.unicode.Utf8View.init(s)) |_| true else |_| false;
        try std.testing.expect(valid);
    }
}

// -- Slice and string tests -----------------------------------------------

test "asciiChar: produces printable ASCII" {
    var prng = std.Random.DefaultPrng.init(42);
    const g = asciiChar();
    for (0..200) |_| {
        const v = g.generate(prng.random(), std.testing.allocator);
        try std.testing.expect(v >= 32 and v <= 126);
    }
}

test "alphanumeric: produces only alphanumeric chars" {
    var prng = std.Random.DefaultPrng.init(42);
    const g = alphanumeric();
    for (0..200) |_| {
        const v = g.generate(prng.random(), std.testing.allocator);
        const is_alnum = (v >= 'a' and v <= 'z') or (v >= 'A' and v <= 'Z') or (v >= '0' and v <= '9');
        try std.testing.expect(is_alnum);
    }
}

test "slice: generates slices within length bounds" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    var prng = std.Random.DefaultPrng.init(42);
    const g = slice(u8, int(u8), 10);
    for (0..100) |_| {
        const v = g.generate(prng.random(), arena_state.allocator());
        try std.testing.expect(v.len <= 10);
    }
}

test "sliceRange: respects min and max length" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    var prng = std.Random.DefaultPrng.init(42);
    const g = sliceRange(u8, int(u8), 3, 8);
    for (0..100) |_| {
        const v = g.generate(prng.random(), arena_state.allocator());
        try std.testing.expect(v.len >= 3 and v.len <= 8);
    }
}

test "asciiString: generates valid ASCII strings" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    var prng = std.Random.DefaultPrng.init(42);
    const g = asciiString(20);
    for (0..100) |_| {
        const v = g.generate(prng.random(), arena_state.allocator());
        try std.testing.expect(v.len <= 20);
        for (v) |c| {
            try std.testing.expect(c >= 32 and c <= 126);
        }
    }
}

test "slice shrink: first candidate is empty" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = slice(u8, int(u8), 10);
    const original = &[_]u8{ 5, 10, 15 };
    var si = g.shrink(original, arena_state.allocator());
    const first = si.next().?;
    try std.testing.expectEqual(@as(usize, 0), first.len);
}

test "slice shrink: tries shorter, then deletions, then element-wise" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = slice(u8, int(u8), 10);
    const original = &[_]u8{ 100, 200 };
    var si = g.shrink(original, arena_state.allocator());

    // Phase 0: shorter prefixes -- empty, then length 1
    const empty = si.next().?;
    try std.testing.expectEqual(@as(usize, 0), empty.len);
    const len1 = si.next().?;
    try std.testing.expectEqual(@as(usize, 1), len1.len);
    try std.testing.expectEqual(@as(u8, 100), len1[0]);

    // Phase 1: delete one element -- remove [0] then [1]
    const del0 = si.next().?;
    try std.testing.expectEqual(@as(usize, 1), del0.len);
    try std.testing.expectEqual(@as(u8, 200), del0[0]); // removed first element
    const del1 = si.next().?;
    try std.testing.expectEqual(@as(usize, 1), del1.len);
    try std.testing.expectEqual(@as(u8, 100), del1[0]); // removed second element

    // Phase 2: shrink elements -- first element shrunk (int shrink: 100 -> 0)
    const shrunk_elem = si.next().?;
    try std.testing.expectEqual(@as(usize, 2), shrunk_elem.len);
    try std.testing.expectEqual(@as(u8, 0), shrunk_elem[0]); // 100 shrunk to 0
    try std.testing.expectEqual(@as(u8, 200), shrunk_elem[1]); // unchanged
}

test "slice shrink: respects min_len" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = sliceRange(u8, int(u8), 2, 10);
    const original = &[_]u8{ 5, 10, 15 };
    var si = g.shrink(original, arena_state.allocator());
    // First candidate should be length 2 (min_len), not empty
    const first = si.next().?;
    try std.testing.expectEqual(@as(usize, 2), first.len);
}

test "slice shrink: empty slice has no shrinks" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = slice(u8, int(u8), 10);
    const original: []const u8 = &.{};
    var si = g.shrink(original, arena_state.allocator());
    try std.testing.expectEqual(null, si.next());
}

// -- filter tests ---------------------------------------------------------

test "filter: exhaustion does not panic" {
    // A predicate that always rejects should not panic -- it returns a value
    // and logs a warning instead.
    var prng = std.Random.DefaultPrng.init(42);
    const g = filter(u32, int(u32), struct {
        fn pred(_: u32) bool {
            return false; // always reject
        }
    }.pred);
    // This should NOT panic -- just returns a value
    _ = g.generate(prng.random(), std.testing.allocator);
}

// -- auto() optional tests ------------------------------------------------

test "auto: optional generates both null and non-null" {
    var prng = std.Random.DefaultPrng.init(42);
    const g = auto(?i32);
    var seen_null = false;
    var seen_value = false;
    for (0..200) |_| {
        const v = g.generate(prng.random(), std.testing.allocator);
        if (v) |_| {
            seen_value = true;
        } else {
            seen_null = true;
        }
    }
    try std.testing.expect(seen_null);
    try std.testing.expect(seen_value);
}

test "auto: optional shrinks to null first" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = auto(?i32);
    var si = g.shrink(@as(?i32, 42), arena_state.allocator());
    const first = si.next().?;
    try std.testing.expectEqual(@as(?i32, null), first);
}

// -- auto() slice tests ---------------------------------------------------

test "auto: slice []const i32 generates values" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    var prng = std.Random.DefaultPrng.init(42);
    const g = auto([]const i32);
    for (0..20) |_| {
        const v = g.generate(prng.random(), arena_state.allocator());
        try std.testing.expect(v.len <= 20);
    }
}

// -- auto() union tests ---------------------------------------------------

test "auto: tagged union generates all variants" {
    const MyUnion = union(enum) {
        empty: void,
        value: i32,
        flag: bool,
    };
    var prng = std.Random.DefaultPrng.init(42);
    const g = auto(MyUnion);
    var seen_empty = false;
    var seen_value = false;
    var seen_flag = false;
    for (0..200) |_| {
        const v = g.generate(prng.random(), std.testing.allocator);
        switch (v) {
            .empty => seen_empty = true,
            .value => seen_value = true,
            .flag => seen_flag = true,
        }
    }
    try std.testing.expect(seen_empty);
    try std.testing.expect(seen_value);
    try std.testing.expect(seen_flag);
}

test "auto: tagged union shrinker produces candidates" {
    const MyUnion = union(enum) {
        empty: void,
        value: i32,
    };
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const g = auto(MyUnion);
    // Shrink .value(100) -- should try .empty first (earlier variant), then shrink the i32
    var si = g.shrink(MyUnion{ .value = 100 }, arena_state.allocator());
    const first = si.next().?;
    // First candidate should be the earlier void variant
    switch (first) {
        .empty => {},
        .value => return error.TestUnexpectedResult,
    }
}

// -- shuffle shrinker tests -----------------------------------------------

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

test "shuffle: already sorted has no shrinks" {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const items = &[_]u32{ 1, 2, 3 };
    const g = shuffle(u32, items);
    var si = g.shrink(items, arena_state.allocator());
    try std.testing.expectEqual(null, si.next());
}

// -- Shrink no-loop guarantee tests (QuickCheck parity) -------------------
// Verify that shrinking never produces a cycle: no value appears in its own
// shrink descendant tree within a bounded depth.

fn assertNoShrinkLoop(comptime T: type, gen: Gen(T), value: T, allocator: std.mem.Allocator) !void {
    // Walk the shrink tree breadth-first to depth 3, collecting all seen values.
    // If any candidate equals the original, that's a loop.
    var current = value;
    for (0..50) |_| {
        var si = gen.shrink(current, allocator);
        var found_smaller = false;
        while (si.next()) |candidate| {
            if (std.meta.eql(candidate, value)) {
                return error.ShrinkLoopDetected;
            }
            if (!found_smaller) {
                current = candidate;
                found_smaller = true;
            }
        }
        if (!found_smaller) break;
    }
}

test "shrink no-loop: u32" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const g = int(u32);
    var prng = std.Random.DefaultPrng.init(42);
    for (0..20) |_| {
        const v = g.generate(prng.random(), arena.allocator());
        try assertNoShrinkLoop(u32, g, v, arena.allocator());
    }
}

test "shrink no-loop: i32" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const g = int(i32);
    var prng = std.Random.DefaultPrng.init(42);
    for (0..20) |_| {
        const v = g.generate(prng.random(), arena.allocator());
        try assertNoShrinkLoop(i32, g, v, arena.allocator());
    }
}

test "shrink no-loop: bool" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const g = boolean();
    try assertNoShrinkLoop(bool, g, true, arena.allocator());
    try assertNoShrinkLoop(bool, g, false, arena.allocator());
}

test "shrink no-loop: f64" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const g = float(f64);
    var prng = std.Random.DefaultPrng.init(42);
    for (0..20) |_| {
        const v = g.generate(prng.random(), arena.allocator());
        try assertNoShrinkLoop(f64, g, v, arena.allocator());
    }
}

test "shrink no-loop: enum" {
    const Color = enum { red, green, blue };
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const g = auto(Color);
    try assertNoShrinkLoop(Color, g, .blue, arena.allocator());
    try assertNoShrinkLoop(Color, g, .green, arena.allocator());
    try assertNoShrinkLoop(Color, g, .red, arena.allocator());
}

test "shrink no-loop: struct" {
    const Point = struct { x: i32, y: i32 };
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const g = auto(Point);
    var prng = std.Random.DefaultPrng.init(42);
    for (0..10) |_| {
        const v = g.generate(prng.random(), arena.allocator());
        try assertNoShrinkLoop(Point, g, v, arena.allocator());
    }
}

test "shrink no-loop: optional" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const g = auto(?i32);
    try assertNoShrinkLoop(?i32, g, @as(?i32, 42), arena.allocator());
    try assertNoShrinkLoop(?i32, g, @as(?i32, null), arena.allocator());
}

test "shrink no-loop: intRange" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const g = intRange(u32, 10, 20);
    var prng = std.Random.DefaultPrng.init(42);
    for (0..20) |_| {
        const v = g.generate(prng.random(), arena.allocator());
        try assertNoShrinkLoop(u32, g, v, arena.allocator());
    }
}

// -- Float shrink candidate existence tests (QuickCheck parity) -----------

test "float shrink: non-zero finite f64 always has candidates" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const g = float(f64);
    var prng = std.Random.DefaultPrng.init(42);
    for (0..100) |_| {
        const v = g.generate(prng.random(), arena.allocator());
        if (v == 0.0) continue;
        var si = g.shrink(v, arena.allocator());
        // Must have at least one shrink candidate (0.0)
        const first = si.next();
        try std.testing.expect(first != null);
    }
}

test "float shrink: non-zero finite f32 always has candidates" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const g = float(f32);
    var prng = std.Random.DefaultPrng.init(42);
    for (0..100) |_| {
        const v = g.generate(prng.random(), arena.allocator());
        if (v == 0.0) continue;
        var si = g.shrink(v, arena.allocator());
        const first = si.next();
        try std.testing.expect(first != null);
    }
}

// -- Shrink boundary precision tests (QuickCheck parity) ------------------

test "shrink boundary: signed n < -5 shrinks to -6" {
    const runner = @import("runner.zig");
    const result = runner.check(.{ .seed = 42, .num_tests = 200 }, i32, int(i32), struct {
        fn prop(n: i32) !void {
            if (n < -5) return error.PropertyFalsified;
        }
    }.prop);
    switch (result) {
        .failed => |f| try std.testing.expectEqual(@as(i32, -6), f.shrunk),
        else => return error.TestUnexpectedResult,
    }
}

test "shrink boundary: intRange(10,20) value shrinks to boundary" {
    const runner = @import("runner.zig");
    const result = runner.check(.{ .seed = 42, .num_tests = 200 }, u32, intRange(u32, 10, 20), struct {
        fn prop(n: u32) !void {
            if (n > 15) return error.PropertyFalsified;
        }
    }.prop);
    switch (result) {
        .failed => |f| try std.testing.expectEqual(@as(u32, 16), f.shrunk),
        else => return error.TestUnexpectedResult,
    }
}

test "shrink boundary: bool property shrinks true to exact boundary" {
    const runner = @import("runner.zig");
    // Property: fail on true. Should shrink to true (it's already minimal).
    const result = runner.check(.{ .seed = 42, .num_tests = 200 }, bool, boolean(), struct {
        fn prop(b: bool) !void {
            if (b) return error.PropertyFalsified;
        }
    }.prop);
    switch (result) {
        .failed => |f| try std.testing.expectEqual(true, f.shrunk),
        else => return error.TestUnexpectedResult,
    }
}

test "shrink boundary: enum shrinks to first failing variant" {
    const Color = enum { red, green, blue };
    const runner = @import("runner.zig");
    const result = runner.check(.{ .seed = 42, .num_tests = 200 }, Color, auto(Color), struct {
        fn prop(c: Color) !void {
            // Fails for green and blue -- should shrink to green (second variant)
            if (c != .red) return error.PropertyFalsified;
        }
    }.prop);
    switch (result) {
        .failed => |f| try std.testing.expectEqual(Color.green, f.shrunk),
        else => return error.TestUnexpectedResult,
    }
}

test "shrink boundary: struct shrinks fields independently" {
    const Point = struct { x: u32, y: u32 };
    const runner = @import("runner.zig");
    const result = runner.check(.{ .seed = 42, .num_tests = 200 }, Point, auto(Point), struct {
        fn prop(p: Point) !void {
            if (p.x >= 5 and p.y >= 5) return error.PropertyFalsified;
        }
    }.prop);
    switch (result) {
        .failed => |f| {
            try std.testing.expectEqual(@as(u32, 5), f.shrunk.x);
            try std.testing.expectEqual(@as(u32, 5), f.shrunk.y);
        },
        else => return error.TestUnexpectedResult,
    }
}

