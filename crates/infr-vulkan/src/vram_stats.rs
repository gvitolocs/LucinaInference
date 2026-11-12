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
//! THE COMMITTED VIEW. Because the logical rows cannot add up to what the driver has committed —
//! and a refusal that says "23.91 GiB already in use" against rows summing to 20.2 GiB is exactly
//! the moment that gap has to be attributed — the same line also carries [`VramCommitted`], the
//! COMMITTED counterpart, read from the two parties that actually commit bytes:
//!
//!   * the driver's live `VK_EXT_memory_budget` per-heap `heapUsage` (the figure the budget guard
//!     already budgets against), and
//!   * gpu-allocator's own report — bytes reserved in `vkAllocateMemory` blocks vs bytes handed out
//!     inside them, plus the largest live allocations by name.
//!
//! Between them sits [`UntrackedCharge`]: device memory this backend holds that carries no
//! [`VramCharge`] and therefore appears in NO usage row — the resident-BDA weight arena's blocks
//! (net of the weight bytes carved out of them, which the `weights` row does see), the staging
//! ring, the transient upload/download staging buffers and the P2P export/import allocations. The
//! line is then a ledger the reader can check by eye:
//!
//! ```text
//! logical + untracked + allocator slack  ≈  driver heap usage
//! ```
//!
//! with the remainder printed as `unexplained` — other processes, the driver's own internals
//! (descriptor pools, command buffers, pipeline/shader memory, page-table and BO metadata: none of
//! it goes through gpu-allocator, so it is in `driver` and nowhere else), and any heap the two
//! sides count differently (the allocator's report spans EVERY memory type; `driver` spans only the
//! heaps the guard budgets). It is a LOWER BOUND: an untracked allocation that is itself an
//! allocator block is subtracted from `reserved` twice, as [`VramCommitted::unexplained`] spells
//! out. A large `unexplained` is the finding, not a formatting failure.
//!
//! The committed device-local total the guard budgets against stays `VulkanShared::device_used`'s
//! job; the guard now prints that total, this breakdown and the committed view next to each other.

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

/// How many `untracked` labels the one-line report lists, and how many allocator allocation NAMES
/// the committed view lists. Both are "the busiest few", and the aggregate they sit under is the
/// real number — these are only there to say WHERE the bytes are.
const TOP_LABELS: usize = 6;
const TOP_NAMES: usize = 4;

/// The `untracked` label for a resident-BDA weight arena block — the dedicated allocation
/// `VulkanBackend::bda_weight_alloc` opens when a weight tensor outgrows every block already open.
/// Each tensor carved out of it hands its own logical bytes back to this tally (see
/// [`VramCounters::charge_carve`]), so the row nets down to what the `weights` row cannot see: the
/// block's stranded tail and its per-tensor alignment padding.
pub const UNTRACKED_ARENA: &str = "arena-block";

/// The label for the weight-upload staging ring's slots (4 × 32 MiB, `MemoryLocation::CpuToGpu`),
/// which no `BufferUsage` ever claims. On a ReBAR card they are VRAM; on a limited-bar device they
/// are system RAM — either way they are the allocator's bytes, not any class's.
pub const UNTRACKED_STAGING_RING: &str = "staging-ring";

/// The labels for the one-shot staging buffers `Backend::upload`/`Backend::download` allocate
/// outside the ring (the non-weight-load paths). Transient by construction, so they show up in a
/// line only when a refusal lands in the middle of one.
pub const UNTRACKED_UPLOAD: &str = "upload_staging";
pub const UNTRACKED_DOWNLOAD: &str = "download_staging";

/// The label for a P2P export or import (`Backing::External`) — a dedicated `vkAllocateMemory`
/// outside gpu-allocator, deliberately outside the budget accounting (see `p2p.rs`).
pub const UNTRACKED_P2P: &str = "p2p";

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
        // SATURATING, not a bare `fetch_sub`: these counters are observation only, and a token
        // released twice (or a carve with no matching block) would otherwise wrap live to ~18 EiB
        // and print that as a fact — the one failure mode a diagnostic ledger must not have.
        let mut live = self.live.load(Ordering::Relaxed);
        loop {
            let next = live.saturating_sub(bytes);
            match self
                .live
                .compare_exchange_weak(live, next, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => break,
                Err(current) => live = current,
            }
        }
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

/// One live allocation's outstanding charge AGAINST THE UNTRACKED TALLY — the COMMITTED bytes of
/// device memory that no [`VramCharge`] classifies. Held by the owning `VkBuffer` and undone by this
/// type's `Drop`, exactly like [`VramCharge`], and for the same reason: a freed allocation leaves the
/// tally without the free site having to remember. Deliberately NOT `Clone`.
#[derive(Debug)]
pub struct UntrackedCharge {
    counters: Arc<VramCounters>,
    label: &'static str,
    bytes: u64,
    /// `false` — the ordinary flavour, [`VramCounters::charge_untracked`]: `bytes` of COMMITTED
    /// device memory that no [`VramCharge`] ever claims, added now and released on drop.
    ///
    /// `true` — [`VramCounters::charge_carve`]: `bytes` carved out of an already-untracked
    /// allocation (a resident-BDA weight out of its arena block) that a `VramCharge` DOES claim, so
    /// they leave the untracked tally for as long as the sub-allocation lives and return to it when
    /// it dies. Without this half the arena's blocks would count twice — once here at their full
    /// committed size, once under `weights` — and `logical + untracked` would overshoot the driver
    /// figure by the whole weight set instead of explaining what it misses.
    ///
    /// Net effect either way: the untracked row is exactly what the usage rows are NOT.
    carve: bool,
}

impl Drop for UntrackedCharge {
    fn drop(&mut self) {
        let pair = self.counters.untracked_pair(self.label);
        if self.carve {
            pair.add(self.bytes);
        } else {
            pair.sub(self.bytes);
        }
    }
}

/// Live + peak bytes per usage class, plus the same pair per scratch-pool tag and per untracked
/// allocation label.
///
/// All arithmetic is atomic, since one backend (and therefore one set of these counters) is shared
/// across threads: the engine records forwards from worker threads, and a multi-device backend
/// hands the same table to every rank. The tag map is keyed by the `&'static str` literals the
/// scratch pools use, so its size is bounded by the number of distinct tags, not by traffic. The
/// untracked map likewise by the handful of `UNTRACKED_*` label constants.
#[derive(Debug, Default)]
pub struct VramCounters {
    usage: [BytePair; USAGE_SLOTS],
    tags: Mutex<HashMap<&'static str, Arc<BytePair>>>,
    /// COMMITTED bytes of the device allocations that carry no `VramCharge`, by label — see
    /// [`UntrackedCharge`]. Keyed by label rather than summed straight into one atomic because "the
    /// arena blocks" and "the staging ring" are different answers to the same question.
    untracked: Mutex<HashMap<&'static str, Arc<BytePair>>>,
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

    /// Record `bytes` of COMMITTED device memory that no [`BufferUsage`] class will ever claim —
    /// an arena block, a staging slot, a P2P import — under `label`, and return the token that
    /// releases them. The counterpart of [`charge`](Self::charge) for the part of the device the
    /// usage rows are blind to.
    pub fn charge_untracked(self: &Arc<Self>, label: &'static str, bytes: u64) -> UntrackedCharge {
        self.untracked_pair(label).add(bytes);
        UntrackedCharge {
            counters: Arc::clone(self),
            label,
            bytes,
            carve: false,
        }
    }

    /// Record `bytes` that a [`charge`](Self::charge) DOES claim but that sit INSIDE an allocation
    /// already charged by [`charge_untracked`](Self::charge_untracked) — the resident-BDA weight
    /// tensor and its arena block. They leave `label`'s untracked tally for as long as the token
    /// lives, so the two rows never both count the same byte.
    pub fn charge_carve(self: &Arc<Self>, label: &'static str, bytes: u64) -> UntrackedCharge {
        self.untracked_pair(label).sub(bytes);
        UntrackedCharge {
            counters: Arc::clone(self),
            label,
            bytes,
            carve: true,
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

    /// The pair for an untracked label, created on first use.
    fn untracked_pair(&self, label: &'static str) -> Arc<BytePair> {
        Arc::clone(
            self.untracked
                .lock()
                .unwrap()
                .entry(label)
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

    /// A point-in-time copy of every counter — the COMMITTED view stays unset here (it needs the
    /// device to ask); [`crate::VulkanBackend::vram_breakdown`] fills it.
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
        let untracked = self
            .untracked
            .lock()
            .unwrap()
            .iter()
            .map(|(name, pair)| ((*name).to_string(), pair.live(), pair.peak()))
            .collect();
        VramBreakdown::sorted(usage, tags, untracked)
    }

    /// [`snapshot`](Self::snapshot)'s one-line rendering.
    pub fn breakdown_line(&self) -> String {
        self.snapshot().line()
    }
}

/// A point-in-time copy of [`VramCounters`] plus, when a device was there to ask, the committed
/// view of the same memory: `usage → (live, peak)`, `tag → (live, peak)`, `label → (live, peak)`
/// for the untracked allocations, and [`VramCommitted`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VramBreakdown {
    /// `(class, live, peak)` — every class in the report order when the snapshot came from a live
    /// backend, so a class that never allocated still shows as `0.0/0.0` rather than vanishing.
    pub usage: Vec<(BufferUsage, u64, u64)>,
    /// `(tag, live, peak)` — every scratch-pool tag seen, highest peak first.
    pub tags: Vec<(String, u64, u64)>,
    /// `(label, live, peak)` — every [`UntrackedCharge`] label seen, highest peak first. COMMITTED
    /// bytes, unlike the logical `usage` rows, and the part of the device no class claims.
    pub untracked: Vec<(String, u64, u64)>,
    /// What the driver and gpu-allocator say is actually reserved. `None` on a counters-only
    /// snapshot — [`VramCounters::snapshot`] has no device to ask,
    /// [`crate::VulkanBackend::vram_breakdown`] does — and a `None` prints no committed segment
    /// at all.
    pub committed: Option<VramCommitted>,
}

/// One heap's committed figures, straight from the driver: `VK_EXT_memory_budget`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HeapCommitted {
    /// Vulkan heap index.
    pub index: u32,
    /// `VK_MEMORY_HEAP_DEVICE_LOCAL_BIT` — on a discrete card this is the VRAM heap, on an APU both
    /// heaps are the same DDR (and the guard budgets against both, see
    /// [`crate::VulkanBackend::vram`]'s `uma` flag).
    pub device_local: bool,
    /// Bytes in use on this heap right now, by EVERY process.
    pub usage: u64,
    /// The ceiling this process may reach on this heap — a budget, not free bytes.
    pub budget: u64,
    /// The heap's physical size.
    pub size: u64,
}

/// The COMMITTED side of the ledger: bytes the driver and gpu-allocator have really reserved, as
/// opposed to the logical sizes [`VramCounters`] hands out. Built by
/// [`crate::VulkanBackend::vram_committed`] from one driver query plus one allocator report.
///
/// The three figures answer three different questions and are deliberately NOT interchangeable:
/// `driver` is everything in use on the guarded heaps (other processes and the driver's own
/// internals included), `reserved` is what THIS process's gpu-allocator blocks cost, `allocated` is
/// the part of those blocks a live allocation actually occupies.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VramCommitted {
    /// The driver's committed bytes across the heaps the budget guard counts — summed
    /// `heapUsage`, the raw figure rather than the guard's clamped `total − available` (they differ
    /// only once something has oversubscribed). When `driver_live` is false this is `tracked`
    /// instead, because there is no extension to ask.
    pub driver: u64,
    /// True when `driver` came from `VK_EXT_memory_budget`.
    pub driver_live: bool,
    /// `VulkanShared::device_used`: device-local COMMITTED bytes this backend has charged to the
    /// budget guard — its own tally, and the figure the guard falls back on when the extension is
    /// absent.
    pub tracked: u64,
    /// gpu-allocator: bytes reserved by its `vkAllocateMemory` blocks (dedicated allocations
    /// included — the vendored 0.27 `generate_report` counts them as one-allocation blocks), and
    /// the bytes of those handed out to live allocations.
    pub reserved: u64,
    pub allocated: u64,
    /// Live blocks behind `reserved`.
    pub blocks: usize,
    /// Per-heap driver figures, in Vulkan heap order, only for the heaps the guard counts.
    pub heaps: Vec<HeapCommitted>,
    /// The allocator's biggest LIVE allocations grouped by name — `(name, committed bytes, how
    /// many)` — biggest first. The names are the `label` each `make_buf_ex` passed, so this is the
    /// committed counterpart of the usage rows: it says which allocator labels hold `allocated`.
    pub names: Vec<(String, u64, usize)>,
}

impl VramCommitted {
    /// Bytes reserved in blocks that no live allocation occupies: `reserved − allocated`. Block
    /// rounding, freed ranges nothing has reused yet, and the general blocks gpu-allocator retains
    /// (it frees an emptied one only while another general block exists in the same memory type).
    pub fn slack(&self) -> u64 {
        self.reserved.saturating_sub(self.allocated)
    }

    /// `driver − (reserved + untracked)`: what neither this process's allocator blocks nor its
    /// untracked allocations explain. SIGNED on purpose — a negative remainder is itself the
    /// finding (gpu-allocator's report spans EVERY memory type while `driver` spans only the heaps
    /// the guard budgets, so a big host-visible block set on a discrete card legitimately shows up
    /// here), and saturating it to 0 would hide that.
    ///
    /// READ IT AS A LOWER BOUND. An untracked arena block, ring slot or transient IS itself an
    /// allocator block, so those bytes sit in `reserved` and get subtracted a second time here; the
    /// remainder is low by exactly the untracked bytes that came from gpu-allocator. The carve
    /// tokens keep the arena's share of that bias to its stranded tail plus padding, so what is
    /// double-subtracted is the staging ring (~128 MiB on a ReBAR card) and a live upload's
    /// transient — not gigabytes.
    pub fn unexplained(&self, untracked: u64) -> i64 {
        let ours = self.reserved.saturating_add(untracked);
        if self.driver >= ours {
            (self.driver - ours) as i64
        } else {
            -((ours - self.driver) as i64)
        }
    }
}

/// MiB with one decimal, SIGNED — for [`VramCommitted::unexplained`], whose sign is the message.
fn mib_signed(bytes: i64) -> String {
    format!("{:.1}", bytes as f64 / (1 << 20) as f64)
}

impl VramBreakdown {
    /// The tag list in report order: biggest PEAK first, ties alphabetical so two runs of the same
    /// numbers print the same line.
    fn sort_tags(tags: &mut [(String, u64, u64)]) {
        tags.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
    }

    fn sorted(
        usage: Vec<(BufferUsage, u64, u64)>,
        tags: Vec<(String, u64, u64)>,
        untracked: Vec<(String, u64, u64)>,
    ) -> Self {
        let mut tags = tags;
        let mut untracked = untracked;
        Self::sort_tags(&mut tags);
        Self::sort_tags(&mut untracked);
        Self {
            usage,
            tags,
            untracked,
            committed: None,
        }
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

    /// `(live, peak)` of one untracked label, or zeros when nothing carries that label.
    pub fn untracked(&self, label: &str) -> (u64, u64) {
        self.untracked
            .iter()
            .find(|(t, _, _)| t == label)
            .map_or((0, 0), |(_, live, peak)| (*live, *peak))
    }

    /// Total COMMITTED bytes held by allocations with no `VramCharge` — the `untracked U` of the
    /// ledger's `logical + untracked + slack ≈ driver`.
    pub fn untracked_live(&self) -> u64 {
        self.untracked.iter().map(|(_, live, _)| *live).sum()
    }

    /// True when nothing has been charged to this snapshot at all — the state of a backend that
    /// exists but has not allocated.
    pub fn is_empty(&self) -> bool {
        self.usage
            .iter()
            .all(|(_, live, peak)| *live == 0 && *peak == 0)
            && self.tags.is_empty()
            && self.untracked.is_empty()
    }

    /// THE one-line report: `MiB live/peak` per usage class — all six, in `USAGE_ORDER` — then the
    /// eight busiest scratch-pool tags by peak (`TOP_TAGS`), the busiest untracked labels
    /// (`TOP_LABELS`) and, when a device was there to ask, the committed ledger
    /// ([`VramCommitted`]).
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
        if !self.untracked.is_empty() {
            let labels = self
                .untracked
                .iter()
                .take(TOP_LABELS)
                .map(|(t, live, peak)| format!("{t} {}/{}", mib(*live), mib(*peak)))
                .collect::<Vec<_>>()
                .join(", ");
            line.push_str("; untracked (MiB live/peak): ");
            line.push_str(&labels);
        }
        if let Some(c) = &self.committed {
            let untracked = self.untracked_live();
            let mut seg = format!(
                "committed: driver heap {} MiB, allocator reserved {} MiB (allocated {}), slack \
                 {}, untracked {}, unexplained {}",
                mib(c.driver),
                mib(c.reserved),
                mib(c.allocated),
                mib(c.slack()),
                mib(untracked),
                mib_signed(c.unexplained(untracked)),
            );
            // Say so when `driver` is not the driver's: without the marker a reader compares this
            // process's own tally against another run's live figure.
            if !c.driver_live {
                seg.push_str(
                    " [driver figure: this process's tracked bytes, no \
                              VK_EXT_memory_budget]",
                );
            }
            seg.push_str(&format!("; blocks {}", c.blocks));
            // Only worth the bytes when there is more than one heap to tell apart — with one, the
            // `driver heap` figure above IS that heap.
            if c.heaps.len() > 1 {
                let heaps = c
                    .heaps
                    .iter()
                    .map(|h| {
                        format!(
                            "{} {} usage/budget {}/{}",
                            h.index,
                            if h.device_local { "dev-local" } else { "host" },
                            mib(h.usage),
                            mib(h.budget)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                seg.push_str("; heaps: ");
                seg.push_str(&heaps);
            }
            if !c.names.is_empty() {
                let names = c
                    .names
                    .iter()
                    .take(TOP_NAMES)
                    .map(|(n, bytes, count)| format!("{n} {}×{count}", mib(*bytes)))
                    .collect::<Vec<_>>()
                    .join(", ");
                seg.push_str("; allocator top: ");
                seg.push_str(&names);
            }
            line.push_str("; ");
            line.push_str(&seg);
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
            untracked: Vec::new(),
            committed: None,
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
            ..VramBreakdown::default()
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
        // A counters-only snapshot has no device to ask, so it prints no committed ledger at all —
        // the segment appears ONLY when `VramBreakdown::committed` is filled.
        assert!(!line.contains("committed:"), "{line}");
        assert!(!line.contains("untracked"), "{line}");
    }

    /// The untracked tally is RAII like the usage tally, and a CARVE is its mirror image: the
    /// arena block's committed bytes go in once, every weight tensor carved out of them takes its
    /// own logical bytes back out for as long as it lives, and both come back to the block when the
    /// block is freed. This is the arithmetic that keeps `weights` + `untracked` from counting the
    /// same 16 GiB twice.
    #[test]
    fn vram_untracked_carves_net_against_their_block() {
        let mib = |n: u64| n << 20;
        let c = counters();

        // One 64 MiB arena block, then three tensors carved out of it (8 + 8 + 16 logical MiB).
        let block = c.charge_untracked(UNTRACKED_ARENA, mib(64));
        let mut carves = Vec::new();
        for n in [8u64, 8, 16] {
            carves.push(c.charge_carve(UNTRACKED_ARENA, mib(n)));
        }
        let snap = c.snapshot();
        assert_eq!(
            snap.untracked(UNTRACKED_ARENA),
            (mib(32), mib(64)),
            "64 committed minus 32 carved, with the block's own size as the peak"
        );
        assert_eq!(snap.untracked_live(), mib(32));

        // A tensor freed hands its bytes back to the untracked tally (the block still holds them).
        drop(carves.remove(0));
        assert_eq!(c.snapshot().untracked(UNTRACKED_ARENA).0, mib(40));

        // The remaining tensors die BEFORE the block, which is the only order an arena can produce
        // in production: a sub-tensor holds an `Arc<BdaBlockHandle>` clone that keeps the block —
        // and with it the block's token — alive until every sub-tensor is gone (see
        // `Backing::BdaSub`). Live walks 40 → 64 → 0 and never has to be clamped.
        drop(carves);
        assert_eq!(c.snapshot().untracked(UNTRACKED_ARENA).0, mib(64));
        drop(block);
        let snap = c.snapshot();
        assert_eq!(
            snap.untracked(UNTRACKED_ARENA),
            (0, mib(64)),
            "live back to zero — no wrap anywhere — peak kept"
        );
        assert_eq!(snap.untracked_live(), 0);
        assert_eq!(
            c.usage_live(BufferUsage::Activations),
            0,
            "the untracked tally never touches a usage class"
        );
    }

    /// The WRONG drop order — a carve outliving the block it was carved from, which an arena cannot
    /// produce (the sub-tensor keeps the block's `Arc` alive) but a future bug elsewhere could. The
    /// tally CLAMPS at zero instead of wrapping: a line that printed `untracked 1818928 MiB` would
    /// be worse than one that prints 0, because the number would look like a finding.
    #[test]
    fn vram_untracked_underflow_saturates_at_zero() {
        let c = counters();
        let block = c.charge_untracked(UNTRACKED_ARENA, 64 << 20);
        let carve = c.charge_carve(UNTRACKED_ARENA, 8 << 20);
        drop(block);
        assert_eq!(
            c.snapshot().untracked(UNTRACKED_ARENA).0,
            0,
            "40 released against 32 live clamps rather than wrapping"
        );
        drop(carve);
        assert_eq!(c.snapshot().untracked(UNTRACKED_ARENA).0, 8 << 20);
    }

    /// An untracked charge ALONE makes a snapshot non-empty: a backend that has allocated nothing
    /// but a staging ring is not an idle backend.
    #[test]
    fn vram_snapshot_lists_untracked_labels_and_sums_them() {
        let mib = |n: u64| n << 20;
        let c = counters();
        let ring = c.charge_untracked(UNTRACKED_STAGING_RING, mib(128));
        let p2p = c.charge_untracked(UNTRACKED_P2P, mib(4));
        let snap = c.snapshot();
        assert!(!snap.is_empty());
        // Highest PEAK first, so the label that matters most leads.
        assert_eq!(
            snap.untracked
                .iter()
                .map(|(l, _, _)| l.as_str())
                .collect::<Vec<_>>(),
            [UNTRACKED_STAGING_RING, UNTRACKED_P2P]
        );
        assert_eq!(snap.untracked_live(), mib(132));
        assert_eq!(snap.get(BufferUsage::Staging), (0, 0), "a separate tally");

        drop((ring, p2p));
        let snap = c.snapshot();
        assert_eq!(snap.untracked_live(), 0);
        assert!(
            !snap.untracked.is_empty(),
            "the label stays listed, at 0.0 — its peak is history"
        );
        assert_eq!(snap.untracked(UNTRACKED_P2P), (0, mib(4)));
    }

    /// The committed ledger's arithmetic: `slack` is the allocator's own reserved-minus-allocated,
    /// `unexplained` is what the allocator blocks plus the untracked row still do not account for,
    /// and it is SIGNED because a negative remainder is a real finding (the allocator's report spans
    /// every memory type, `driver` only the guarded heaps). Nothing here may wrap: a subtracting
    /// counter that underflows prints 18 EiB as a fact.
    #[test]
    fn vram_committed_ledger_arithmetic() {
        let mib = |n: u64| n << 20;
        let c = VramCommitted {
            driver: mib(23910),
            driver_live: true,
            tracked: mib(20480),
            reserved: mib(20500),
            allocated: mib(20480),
            ..VramCommitted::default()
        };
        assert_eq!(c.slack(), mib(20), "reserved minus allocated");
        // logical + untracked + slack ≈ driver: 20480 tracked-class bytes are inside the 20500
        // reserved, the untracked row adds 3390, and 20 MiB of driver bytes belong to nobody.
        assert_eq!(c.unexplained(mib(3390)), mib(20) as i64);
        assert_eq!(c.unexplained(0), mib(3410) as i64);
        // More ours than the driver reports: negative, and NOT clamped to zero.
        assert_eq!(c.unexplained(mib(4000)), -(mib(590) as i64));
        assert_eq!(
            VramCommitted {
                reserved: mib(10),
                allocated: mib(64),
                ..VramCommitted::default()
            }
            .slack(),
            0,
            "a report that ever contradicted itself saturates instead of wrapping"
        );
        assert_eq!(VramCommitted::default().unexplained(0), 0);
    }

    /// EXACT text of the committed + untracked segments of the report line — the ledger
    /// `infr bench` logs and the budget guard appends to its refusal, pinned so a rename or a unit
    /// change fails a test rather than silently changing what a user pastes.
    #[test]
    fn vram_breakdown_line_formats_committed_ledger() {
        let mib = |n: u64| n << 20;
        let b = VramBreakdown {
            usage: vec![
                (BufferUsage::Weights, mib(16400), mib(16400)),
                (BufferUsage::KvCache, mib(2268), mib(2268)),
                (BufferUsage::Activations, mib(1506), mib(1529)),
                (BufferUsage::Staging, 0, mib(64)),
                (BufferUsage::Readback, 0, mib(2)),
                (BufferUsage::HostWeights, 0, 0),
            ],
            tags: Vec::new(),
            untracked: vec![
                (UNTRACKED_ARENA.to_string(), mib(3358), mib(3358)),
                (UNTRACKED_STAGING_RING.to_string(), mib(32), mib(128)),
            ],
            committed: Some(VramCommitted {
                driver: mib(23910),
                driver_live: true,
                tracked: mib(20480),
                reserved: mib(20500),
                allocated: mib(20480),
                blocks: 12,
                heaps: vec![
                    HeapCommitted {
                        index: 0,
                        device_local: true,
                        usage: mib(23910),
                        budget: mib(24576),
                        size: mib(24576),
                    },
                    HeapCommitted {
                        index: 1,
                        device_local: false,
                        usage: mib(512),
                        budget: mib(31457),
                        size: mib(32768),
                    },
                ],
                names: vec![
                    ("resident-bda".to_string(), mib(16400), 4),
                    ("kv-cache".to_string(), mib(2268), 24),
                ],
            }),
        };
        let line = b.line();
        println!("{line}");
        assert_eq!(
            line,
            "vram by usage (MiB live/peak): weights 16400.0/16400.0, kv_cache 2268.0/2268.0, \
             activations 1506.0/1529.0, staging 0.0/64.0, readback 0.0/2.0, host_weights \
             0.0/0.0; untracked (MiB live/peak): arena-block 3358.0/3358.0, staging-ring \
             32.0/128.0; committed: driver heap 23910.0 MiB, allocator reserved 20500.0 MiB \
             (allocated 20480.0), slack 20.0, untracked 3390.0, unexplained 20.0; blocks 12; \
             heaps: 0 dev-local usage/budget 23910.0/24576.0, 1 host usage/budget \
             512.0/31457.0; allocator top: resident-bda 16400.0×4, kv-cache 2268.0×24"
        );
        assert!(
            !line.contains("top pool tags"),
            "no tags charged, so no tag section"
        );

        // ONE heap in scope: the `driver heap` figure above already IS that heap, so the heap list
        // is not repeated.
        let one_heap = VramBreakdown {
            committed: Some(VramCommitted {
                driver: mib(23910),
                driver_live: true,
                reserved: mib(20500),
                allocated: mib(20480),
                heaps: vec![HeapCommitted {
                    index: 0,
                    device_local: true,
                    usage: mib(23910),
                    budget: mib(24576),
                    size: mib(24576),
                }],
                ..VramCommitted::default()
            }),
            ..VramBreakdown::default()
        };
        let line = one_heap.line();
        println!("{line}");
        assert!(!line.contains("heaps:"), "{line}");
        assert!(
            line.ends_with("untracked 0.0, unexplained 3410.0; blocks 0"),
            "{line}"
        );

        // No VK_EXT_memory_budget: the driver slot says where its number came from, because a
        // reader otherwise compares this process's own tally against another run's live figure.
        let no_ext = VramBreakdown {
            committed: Some(VramCommitted {
                driver: mib(100),
                driver_live: false,
                tracked: mib(100),
                ..VramCommitted::default()
            }),
            ..VramBreakdown::default()
        };
        let line = no_ext.line();
        println!("{line}");
        assert!(
            line.contains(
                "driver heap 100.0 MiB, allocator reserved 0.0 MiB (allocated 0.0), slack 0.0, \
                 untracked 0.0, unexplained 100.0 [driver figure: this process's tracked bytes, \
                 no VK_EXT_memory_budget]"
            ),
            "{line}"
        );
    }
}
