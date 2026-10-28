//! Clock access for code that has no `std.Io` of its own.

const std = @import("std");

fn io() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

/// A seed taken from the wall clock, for runs that did not configure one.
pub fn timeSeed() u64 {
    const ns = std.Io.Clock.real.now(io()).nanoseconds;
    return @truncate(@as(u96, @bitCast(ns)));
}

/// Monotonic nanoseconds since an unspecified point; only differences mean anything.
pub fn monotonicNanos() i96 {
    return std.Io.Clock.awake.now(io()).nanoseconds;
}
