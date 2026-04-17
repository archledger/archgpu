/// Universal per-tweak status. Every module that mutates system state is expected to expose
/// a `check_state(ctx, gpus[, form]) -> TweakState` read-only probe so the GUI can disable
/// redundant toggles and the `auto::recommend` engine can avoid recommending actions that
/// are already done.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TweakState {
    /// The tweak is fully applied on this host.
    Applied,
    /// The tweak is compatible with this host but has NOT been applied yet.
    Unapplied,
    /// The tweak doesn't apply to this host at all (for example, an NVIDIA-specific tweak on
    /// an Intel-only machine). The UI should disable the corresponding switch and show
    /// an "unsupported" label rather than an "already applied" badge.
    Incompatible,
}

impl TweakState {
    pub fn is_applied(self) -> bool {
        matches!(self, Self::Applied)
    }
    pub fn is_unapplied(self) -> bool {
        matches!(self, Self::Unapplied)
    }
    pub fn is_incompatible(self) -> bool {
        matches!(self, Self::Incompatible)
    }
}
