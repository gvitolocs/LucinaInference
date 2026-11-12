//! Live + peak VRAM accounting, per [`BufferUsage`] class and per scratch-pool tag.
//!
//! WHY THIS EXISTS. The budget guard refuses a deep-context run with "48.0 MiB requested +
//! 23.69 GiB already in use" and says nothing about WHO holds the 23.69 GiB; `device_used` — the
//! tally behind that figure — is one running total, so neither the refusal nor the gap against
//! another engine's footprint can be attributed. This module keeps the missing breakdown: for
//! every allocation handed out through a `BufferUsage` entry point it tracks the class's live
//! bytes and the high-water mark that figure has reached, and for the pooled scratch allocations
//! it tracks the same pair per scratch-pool tag (`adapter`'s `pooled(..)` helper already names
//! every scratch buffer with one). It changes no placement, sizing or allocation decision —
//! observation only: a caller that holds the backend reads it through
//! [`crate::VulkanBackend::vram_breakdown`], the budget guard appends it to its refusal message,
//! and the seam hands two snapshots of it to `infr bench`.
//!
//! WHAT THE BYTES ARE. The LOGICAL size the caller asked for — the same convention
//! `VulkanBackend::activation_peak` has always used, and the unit the seam's activation reserve is
//! priced against — not the aligned bytes the driver commits. Three consequences worth knowing
//! before reading a line:
//!
//!   * the resident-BDA weight arena places tensors into blocks with a large minimum size, so
//!     `weights` is the model's weight bytes WITHOUT the stranded tail of the last block;
//!   * `staging` and `readback` are host-visible classes and `host_weights` is system RAM by
//!     definition (though on a ReBAR card a `CpuToGpu` allocation does land in VRAM, which is why
//!     they are reported rather than dropped);
//!   * a `kv_cache` buffer spilled to system RAM by `INFR_KV_OVERFLOW` still counts under
//!     `kv_cache`.
//!
//! The committed device-local total the guard budgets against stays `VulkanShared::device_used`'s
//! job; the guard now prints that total and this breakdown next to each other.

use infr_core::backend::BufferUsage;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Every [`BufferUsage`] class, in the order [`VramBreakdown::line`] reports them.
const USAGE_ORDER: [BufferUsage; 6] = [
    BufferUsage::Weights,
    BufferUsage::KvCache,
    BufferUsage::Activations,
    BufferUsage::Staging,
    BufferUsage::Readback,
    BufferUsage::HostWeights,
];

/// Number of counter slots — one per class in [`USAGE_ORDER`].
const USAGE_SLOTS: usize = USAGE_ORDER.len();

/// How many pool tags the one-line report lists.
const TOP_TAGS: usize = 8;

/// The report name of a usage class: the enum's own spelling, so a printed line reads straight
/// against `BufferUsage` with no translation table kept in the reader's head.
fn usage_name(usage: BufferUsage) -> &'static str {
    match usage {
        BufferUsage::Weights => "weights",
        BufferUsage::KvCache => "kv_cache",
        BufferUsage::Activations => "activations",
        BufferUsage::Staging => "staging",
        BufferUsage::Readback => "readback",
        BufferUsage::HostWeights => "host_weights",
    }
}

/// The counter slot of a usage class. An EXHAUSTIVE match on purpose: a new `BufferUsage` variant
/// must be accounted for here (a compile error) instead of defaulting into somebody else's bytes,
/// and `vram_usage_slots_match_the_report_order` pins the match against [`USAGE_ORDER`] so the two
/// lists cannot drift apart silently.
fn usage_slot(usage: BufferUsage) -> usize {
    match usage {
        BufferUsage::Weights => 0,
        BufferUsage::KvCache => 1,
        BufferUsage::Activations => 2,
        BufferUsage::Staging => 3,
        BufferUsage::Readback => 4,
        BufferUsage::HostWeights => 5,
    }
}

/// MiB with one decimal — the unit the report line is labelled with. Fixed rather than adaptive
/// (unlike the guard's `fmt_bytes`) so the two columns of a class are comparable at a glance.
fn mib(bytes: u64) -> String {
    format!("{:.1}", bytes as f64 / (1 << 20) as f64)
}

/// One live/peak byte pair.
///
/// `peak` is a high-water mark and never comes back down — not when the bytes are freed, and not
/// when the class empties: the number that matters for "how much did this shape really cost" is
/// the mark, and a mark that resets on free is not one.
#[derive(Debug, Default)]
pub struct BytePair {
    live: AtomicU64,
    peak: AtomicU64,
}

impl BytePair {
    fn add(&self, bytes: u64) {
        let live = self.live.fetch_add(bytes, Ordering::Relaxed) + bytes;
        self.peak.fetch_max(live, Ordering::Relaxed);
    }

    fn sub(&self, bytes: u64) {
        self.live.fetch_sub(bytes, Ordering::Relaxed);
    }

    /// Bytes of this pair's class concurrently live right now.
    pub fn live(&self) -> u64 {
        self.live.load(Ordering::Relaxed)
    }

    /// High-water mark this pair's live figure has reached.
    pub fn peak(&self) -> u64 {
        self.peak.load(Ordering::Relaxed)
    }
}

/// One live allocation's outstanding charge, held by the buffer that owns the allocation (infr's
/// `VkBuffer`) and undone by this type's `Drop` — so a freed buffer leaves the tally without any
/// free site having to remember to subtract it. Deliberately NOT `Clone`: a copy would release the
/// bytes twice.
#[derive(Debug)]
pub struct VramCharge {
    counters: Arc<VramCounters>,
    usage: BufferUsage,
    bytes: u64,
    /// The scratch-pool tag's pair, when the allocation named one.
    tag: Option<Arc<BytePair>>,
}

impl Drop for VramCharge {
    fn drop(&mut self) {
        self.counters.usage[usage_slot(self.usage)].sub(self.bytes);
        if let Some(tag) = &self.tag {
            tag.sub(self.bytes);
        }
    }
}

/// Live + peak bytes per usage class, plus the same pair per scratch-pool tag.
///
/// All arithmetic is atomic, since one backend (and therefore one set of these counters) is shared
/// across threads: the engine records forwards from worker threads, and a multi-device backend
/// hands the same table to every rank. The tag map is keyed by the `&'static str` literals the
/// scratch pools use, so its size is bounded by the number of distinct tags, not by traffic.
#[derive(Debug, Default)]
pub struct VramCounters {
    usage: [BytePair; USAGE_SLOTS],
    tags: Mutex<HashMap<&'static str, Arc<BytePair>>>,
}

impl VramCounters {
    /// Record `bytes` of `usage` becoming live — and, when `tag` names one, the same bytes under
    /// that scratch-pool tag too — and return the token that releases them.
    pub fn charge(
        self: &Arc<Self>,
        usage: BufferUsage,
        bytes: u64,
        tag: Option<&'static str>,
    ) -> VramCharge {
        self.usage[usage_slot(usage)].add(bytes);
        let tag_pair = tag.map(|name| {
            let pair = self.tag_pair(name);
            pair.add(bytes);
            pair
        });
        VramCharge {
            counters: Arc::clone(self),
            usage,
            bytes,
            tag: tag_pair,
        }
    }

    /// The pair for a scratch-pool tag, created on first use.
    fn tag_pair(&self, tag: &'static str) -> Arc<BytePair> {
        Arc::clone(
            self.tags
                .lock()
                .unwrap()
                .entry(tag)
                .or_insert_with(|| Arc::new(BytePair::default())),
        )
    }

    /// Live bytes of one usage class.
    pub fn usage_live(&self, usage: BufferUsage) -> u64 {
        self.usage[usage_slot(usage)].live()
    }

    /// Peak live bytes of one usage class — the figure
    /// [`infr_core::backend::Backend::activation_peak`] reports for [`BufferUsage::Activations`].
    pub fn usage_peak(&self, usage: BufferUsage) -> u64 {
        self.usage[usage_slot(usage)].peak()
    }

    /// A point-in-time copy of every counter.
    pub fn snapshot(&self) -> VramBreakdown {
        let usage = USAGE_ORDER
            .iter()
            .map(|&u| {
                let pair = &self.usage[usage_slot(u)];
                (u, pair.live(), pair.peak())
            })
            .collect();
        let tags = self
            .tags
            .lock()
            .unwrap()
            .iter()
            .map(|(name, pair)| ((*name).to_string(), pair.live(), pair.peak()))
            .collect();
        VramBreakdown::sorted(usage, tags)
    }

    /// [`snapshot`](Self::snapshot)'s one-line rendering.
    pub fn breakdown_line(&self) -> String {
        self.snapshot().line()
    }
}

/// A point-in-time copy of [`VramCounters`]: `usage → (live, peak)` and `tag → (live, peak)`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VramBreakdown {
    /// `(class, live, peak)` — every class in the report order when the snapshot came from a live
    /// backend, so a class that never allocated still shows as `0.0/0.0` rather than vanishing.
    pub usage: Vec<(BufferUsage, u64, u64)>,
    /// `(tag, live, peak)` — every scratch-pool tag seen, highest peak first.
    pub tags: Vec<(String, u64, u64)>,
}

impl VramBreakdown {
    /// The tag list in report order: biggest PEAK first, ties alphabetical so two runs of the same
    /// numbers print the same line.
    fn sort_tags(tags: &mut [(String, u64, u64)]) {
        tags.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
    }

    fn sorted(usage: Vec<(BufferUsage, u64, u64)>, tags: Vec<(String, u64, u64)>) -> Self {
        let mut tags = tags;
        Self::sort_tags(&mut tags);
        Self { usage, tags }
    }

    /// `(live, peak)` of one usage class, or zeros when this snapshot does not carry it.
    pub fn get(&self, usage: BufferUsage) -> (u64, u64) {
        self.usage
            .iter()
            .find(|(u, _, _)| *u == usage)
            .map_or((0, 0), |(_, live, peak)| (*live, *peak))
    }

    /// `(live, peak)` of one scratch-pool tag, or zeros when that tag never allocated.
    pub fn tag(&self, tag: &str) -> (u64, u64) {
        self.tags
            .iter()
            .find(|(t, _, _)| t == tag)
            .map_or((0, 0), |(_, live, peak)| (*live, *peak))
    }

    /// True when nothing has been charged to this snapshot at all — the state of a backend that
    /// exists but has not allocated.
    pub fn is_empty(&self) -> bool {
        self.usage
            .iter()
            .all(|(_, live, peak)| *live == 0 && *peak == 0)
            && self.tags.is_empty()
    }

    /// THE one-line report: `MiB live/peak` per usage class — all six, in `USAGE_ORDER` — then the
    /// eight busiest scratch-pool tags by peak (`TOP_TAGS`).
    pub fn line(&self) -> String {
        let usages = self
            .usage
            .iter()
            .map(|(u, live, peak)| format!("{} {}/{}", usage_name(*u), mib(*live), mib(*peak)))
            .collect::<Vec<_>>()
            .join(", ");
        let mut line = format!("vram by usage (MiB live/peak): {usages}");
        if !self.tags.is_empty() {
            let tags = self
                .tags
                .iter()
                .take(TOP_TAGS)
                .map(|(t, live, peak)| format!("{t} {}/{}", mib(*live), mib(*peak)))
                .collect::<Vec<_>>()
                .join(", ");
            line.push_str("; top pool tags: ");
            line.push_str(&tags);
        }
        line
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counters() -> Arc<VramCounters> {
        Arc::new(VramCounters::default())
    }

    /// A freed buffer's bytes come back DOWN, the peak they set does not — the pair is a
    /// high-water mark of live bytes, not a sum of everything allocated.
    #[test]
    fn vram_counters_release_on_free_keeps_the_peak() {
        let c = counters();
        let a = c.charge(BufferUsage::Activations, 4 << 20, None);
        let b = c.charge(BufferUsage::Activations, 8 << 20, None);
        assert_eq!(c.usage_live(BufferUsage::Activations), 12 << 20);
        assert_eq!(c.usage_peak(BufferUsage::Activations), 12 << 20);

        drop(a);
        assert_eq!(c.usage_live(BufferUsage::Activations), 8 << 20);
        assert_eq!(
            c.usage_peak(BufferUsage::Activations),
            12 << 20,
            "the peak must survive the free that lowered live"
        );

        drop(b);
        assert_eq!(c.usage_live(BufferUsage::Activations), 0);
        assert_eq!(c.usage_peak(BufferUsage::Activations), 12 << 20);
    }

    /// A higher peak only comes from bytes that were live AT ONCE: charge/free/charge never moves
    /// it past the biggest concurrent set.
    #[test]
    fn vram_counters_peak_is_concurrency_not_accumulation() {
        let c = counters();
        for _ in 0..4 {
            let _one = c.charge(BufferUsage::KvCache, 3 << 20, None);
            assert_eq!(c.usage_peak(BufferUsage::KvCache), 3 << 20);
        }
        let _two = (
            c.charge(BufferUsage::KvCache, 3 << 20, None),
            c.charge(BufferUsage::KvCache, 3 << 20, None),
        );
        assert_eq!(c.usage_peak(BufferUsage::KvCache), 6 << 20);
    }

    /// Each class counts its own bytes and nothing else's — the whole point of the report.
    #[test]
    fn vram_counters_keep_usages_separate() {
        let mib = |n: u64| n << 20;
        let c = counters();
        let w = c.charge(BufferUsage::Weights, mib(1024), None);
        let k = c.charge(BufferUsage::KvCache, mib(2048), None);
        let a = c.charge(BufferUsage::Activations, mib(4), None);
        let s = c.charge(BufferUsage::Staging, mib(1), None);
        let r = c.charge(BufferUsage::Readback, mib(1), None);
        let h = c.charge(BufferUsage::HostWeights, mib(16), None);

        for (usage, want) in [
            (BufferUsage::Weights, 1024),
            (BufferUsage::KvCache, 2048),
            (BufferUsage::Activations, 4),
            (BufferUsage::Staging, 1),
            (BufferUsage::Readback, 1),
            (BufferUsage::HostWeights, 16),
        ] {
            let live = c.usage_live(usage);
            assert_eq!(
                live,
                mib(want),
                "{usage:?} holds {live} bytes, expected {want} MiB"
            );
        }

        // Freeing two classes moves exactly those two.
        drop((w, h));
        assert_eq!(c.usage_live(BufferUsage::Weights), 0);
        assert_eq!(c.usage_live(BufferUsage::HostWeights), 0);
        assert_eq!(c.usage_live(BufferUsage::KvCache), mib(2048), "untouched");
        assert_eq!(c.usage_live(BufferUsage::Activations), mib(4), "untouched");
        assert_eq!(c.usage_peak(BufferUsage::Weights), mib(1024));

        drop((k, a, s, r));
        for usage in USAGE_ORDER {
            assert_eq!(c.usage_live(usage), 0, "{usage:?} must be empty again");
        }
        assert_eq!(c.usage_peak(BufferUsage::KvCache), mib(2048));
    }

    /// Tags are the sub-breakdown of the pooled allocations: same live/peak rules, and a tag is
    /// shared by every buffer allocated under it.
    #[test]
    fn vram_counters_track_pool_tags() {
        let c = counters();
        let big = c.charge(BufferUsage::Activations, 64 << 20, Some("flash_po"));
        let kv = c.charge(BufferUsage::Activations, 8 << 20, Some("kvdeq_k"));
        assert_eq!(c.usage_live(BufferUsage::Activations), 72 << 20);

        // A second buffer under the SAME tag: the tag accumulates, the class accumulates.
        let more = c.charge(BufferUsage::Activations, 8 << 20, Some("kvdeq_k"));
        let snap = c.snapshot();
        assert_eq!(snap.tag("kvdeq_k"), (16 << 20, 16 << 20));
        assert_eq!(snap.tag("flash_po"), (64 << 20, 64 << 20));

        drop(more);
        assert_eq!(c.snapshot().tag("kvdeq_k"), (8 << 20, 16 << 20));
        assert_eq!(
            c.snapshot().get(BufferUsage::Activations).0,
            72 << 20,
            "the freed tag buffer came back out of the class too"
        );
        drop(big);
        drop(kv);
        let snap = c.snapshot();
        // Class live is empty again; the class PEAK is the biggest set that was live at once
        // (64 + 8 + 8), which here equals the two tag peaks because nothing was freed before it.
        assert_eq!(snap.get(BufferUsage::Activations), (0, 80 << 20));
        assert_eq!(snap.tag("flash_po"), (0, 64 << 20));
        assert_eq!(snap.tag("kvdeq_k"), (0, 16 << 20));
    }

    /// Exact text of the report line for a known snapshot — the format `infr bench` and the guard
    /// both print, pinned here so a rename of a class or a unit change fails a test.
    #[test]
    fn vram_breakdown_line_formats_known_snapshot() {
        let mib = |n: u64| n << 20;
        let b = VramBreakdown {
            usage: vec![
                (BufferUsage::Weights, mib(1024), mib(1024)),
                (
                    BufferUsage::KvCache,
                    mib(2048) + mib(512),
                    mib(2048) + mib(512),
                ),
                (BufferUsage::Activations, mib(48), mib(700)),
                (BufferUsage::Staging, mib(64), mib(64)),
                (BufferUsage::Readback, 0, mib(2)),
                (BufferUsage::HostWeights, 0, 0),
            ],
            tags: vec![("flash_po".to_string(), mib(4), mib(64))],
        };
        let line = b.line();
        println!("{line}");
        assert_eq!(
            line,
            "vram by usage (MiB live/peak): weights 1024.0/1024.0, kv_cache 2560.0/2560.0, \
             activations 48.0/700.0, staging 64.0/64.0, readback 0.0/2.0, host_weights 0.0/0.0; \
             top pool tags: flash_po 4.0/64.0"
        );

        // No tags at all: no `top pool tags` section, and nothing else about the line changes.
        let no_tags = VramBreakdown {
            usage: b.usage.clone(),
            tags: Vec::new(),
        };
        assert!(!no_tags.line().contains("top pool tags"));
        assert!(VramBreakdown::default().is_empty());
        assert!(!b.is_empty());
        assert_eq!(b.get(BufferUsage::Activations), (mib(48), mib(700)));
        assert_eq!(VramBreakdown::default().get(BufferUsage::Weights), (0, 0));
    }

    /// At most [`TOP_TAGS`] tags make the line, and the ones that do are the biggest by PEAK.
    #[test]
    fn vram_breakdown_line_keeps_top_tags_by_peak() {
        let c = counters();
        let mut held = Vec::new();
        for (i, tag) in TAGS.iter().enumerate() {
            held.push(c.charge(BufferUsage::Activations, (i as u64 + 1) << 20, Some(*tag)));
        }
        let line = c.breakdown_line();
        println!("{line}");
        let tags = line.split("top pool tags: ").nth(1).expect("tags section");
        let listed: Vec<&str> = tags
            .split(", ")
            .map(|t| t.split(' ').next().unwrap())
            .collect();
        assert_eq!(
            listed,
            ["tag12", "tag11", "tag10", "tag9", "tag8", "tag7", "tag6", "tag5"]
        );
        for cut in &TAGS[..4] {
            assert!(!listed.contains(cut), "{cut} should have been cut");
        }
    }

    /// Every tag name used by [`vram_breakdown_line_keeps_top_tags_by_peak`] — `'static`, because
    /// that is what the scratch pools key themselves with.
    const TAGS: [&str; 12] = [
        "tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7", "tag8", "tag9", "tag10", "tag11",
        "tag12",
    ];

    /// `usage_slot` and `USAGE_ORDER` describe the same six classes: a `BufferUsage` variant added
    /// later breaks the match (compile error) or this test (drift between the two lists), never a
    /// silently missing line.
    #[test]
    fn vram_usage_slots_match_the_report_order() {
        assert_eq!(USAGE_SLOTS, USAGE_ORDER.len());
        let mut seen = [false; USAGE_SLOTS];
        for (i, &usage) in USAGE_ORDER.iter().enumerate() {
            assert_eq!(usage_slot(usage), i, "{usage:?} out of order");
            assert!(!seen[i], "{usage:?} slot reused");
            seen[i] = true;
        }
        assert!(
            seen.iter().all(|s| *s),
            "a class is missing from USAGE_ORDER"
        );
        assert_eq!(usage_name(BufferUsage::KvCache), "kv_cache");
    }

    /// A snapshot carries EVERY class, so a backend that has not allocated prints all six at
    /// `0.0/0.0` with no tag section — the floor every reported line is read against, and the proof
    /// that a class missing from a line means zero rather than "not measured".
    #[test]
    fn vram_snapshot_reports_every_class_even_untouched() {
        let line = VramCounters::default().breakdown_line();
        println!("{line}");
        assert_eq!(
            line,
            "vram by usage (MiB live/peak): weights 0.0/0.0, kv_cache 0.0/0.0, activations \
             0.0/0.0, staging 0.0/0.0, readback 0.0/0.0, host_weights 0.0/0.0"
        );
        assert!(VramCounters::default().snapshot().is_empty());

        let c = counters();
        let _w = c.charge(BufferUsage::Weights, 4 << 20, None);
        let snap = c.snapshot();
        assert!(!snap.is_empty());
        assert_eq!(
            snap.usage.len(),
            USAGE_SLOTS,
            "all six classes still listed"
        );
        assert_eq!(snap.get(BufferUsage::Weights), (4 << 20, 4 << 20));
        assert_eq!(
            snap.get(BufferUsage::KvCache),
            (0, 0),
            "untouched, not omitted"
        );
        let line = snap.line();
        assert!(!line.contains("top pool tags"), "{line}");
    }
}
