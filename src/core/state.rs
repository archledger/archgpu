/// Universal per-tweak status.
///
/// Phase 17 refactor: split the previous binary "Applied / Unapplied" into four states
/// that capture the gap between "config written" and "kernel actually using it":
///
///   Active         — the live running kernel confirms the tweak is in effect. For bootloader
///                    cmdline params this means the relevant `/sys/module/<m>/parameters/<p>`
///                    reports the expected value; for file-only tweaks (wayland env drop-ins,
///                    gaming packages, power modprobe/suspend services) it simply means the
///                    config is correct and no kernel-level probe distinguishes pre/post-reboot.
///   PendingReboot  — config files are written correctly but the running kernel hasn't adopted
///                    the change yet. Only produced by bootloader tweaks: the cmdline source
///                    has the param but `/sys/module/<m>/parameters/<p>` doesn't match — user
///                    needs to reboot (or rebuild initramfs + reboot).
///   Unapplied      — no config written yet.
///   Incompatible   — the tweak doesn't apply to this host at all (NVIDIA-only tweak on an
///                    Intel-only host, or a GPU-aware param set that's empty for this HW).
///
/// The GUI renders these as four distinct states: green "✓ Active" badge (switch disabled),
/// yellow "⟳ Reboot pending" badge (switch disabled), normal Switch, or orange "Unsupported"
/// badge (switch disabled) respectively.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TweakState {
    Active,
    PendingReboot,
    Unapplied,
    Incompatible,
}

impl TweakState {
    pub fn is_active(self) -> bool {
        matches!(self, Self::Active)
    }
    pub fn is_pending_reboot(self) -> bool {
        matches!(self, Self::PendingReboot)
    }
    pub fn is_unapplied(self) -> bool {
        matches!(self, Self::Unapplied)
    }
    pub fn is_incompatible(self) -> bool {
        matches!(self, Self::Incompatible)
    }
}
