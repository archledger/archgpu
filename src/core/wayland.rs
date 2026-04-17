use anyhow::Result;

use crate::core::gpu::GpuInventory;
use crate::core::state::TweakState;
use crate::core::{prime, Context};
use crate::utils::fs_helper::{write_dropin, ChangeReport};

const PROFILE_D_FILE: &str = "99-nvidia-wayland.sh";
const MKINITCPIO_DROPIN_FILE: &str = "nvidia-modules.conf";

const PROFILE_D_CONTENT: &str = "\
# Managed by arch-nvidia-tweaker — do not edit by hand.
# Exports required for Wayland compositors to use the NVIDIA EGL/GBM backend.
export GBM_BACKEND=nvidia-drm
export LIBVA_DRIVER_NAME=nvidia
export __GLX_VENDOR_LIBRARY_NAME=nvidia
";

// `MODULES+=(...)` appends rather than overwrites (mkinitcpio sources drop-ins after the main
// config; using `=` would clobber the user's existing MODULES line).
const MKINITCPIO_DROPIN_CONTENT: &str = "\
# Managed by arch-nvidia-tweaker — do not edit by hand.
MODULES+=(nvidia nvidia_modeset nvidia_uvm nvidia_drm)
";

/// Returns `Applied` only when BOTH the /etc/profile.d env drop-in AND the mkinitcpio modules
/// drop-in exist on disk. Incompatible on non-NVIDIA hosts.
pub fn check_state(ctx: &Context, gpus: &GpuInventory) -> TweakState {
    if !gpus.has_nvidia() {
        return TweakState::Incompatible;
    }
    let profile = ctx.paths.profile_d.join(PROFILE_D_FILE);
    let mkinit = ctx.paths.mkinitcpio_d.join(MKINITCPIO_DROPIN_FILE);
    if profile.exists() && mkinit.exists() {
        TweakState::Applied
    } else {
        TweakState::Unapplied
    }
}

pub fn apply(ctx: &Context, gpus: &GpuInventory) -> Result<Vec<ChangeReport>> {
    let dry_run = ctx.mode.is_dry_run();
    let mut reports = Vec::with_capacity(3);

    let profile_target = ctx.paths.profile_d.join(PROFILE_D_FILE);
    reports.push(write_dropin(
        &profile_target,
        PROFILE_D_CONTENT,
        &ctx.paths.backup_dir,
        dry_run,
    )?);

    let mkinitcpio_target = ctx.paths.mkinitcpio_d.join(MKINITCPIO_DROPIN_FILE);
    reports.push(write_dropin(
        &mkinitcpio_target,
        MKINITCPIO_DROPIN_CONTENT,
        &ctx.paths.backup_dir,
        dry_run,
    )?);

    // For hybrid (Optimus / PRIME) setups, also drop a PRIME outputclass Xorg config when the
    // default shipped by nvidia-utils isn't present. Pure Wayland sessions ignore the file.
    reports.push(prime::apply(ctx, gpus)?);

    Ok(reports)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::gpu::{GpuInfo, GpuVendor};
    use crate::core::ExecutionMode;
    use tempfile::tempdir;

    fn empty_gpus() -> GpuInventory {
        GpuInventory { gpus: Vec::new() }
    }

    fn single_nvidia() -> GpuInventory {
        GpuInventory {
            gpus: vec![GpuInfo {
                vendor: GpuVendor::Nvidia,
                vendor_id: 0x10de,
                device_id: 0x2684,
                pci_address: "0000:01:00.0".into(),
                vendor_name: "NVIDIA".into(),
                product_name: "RTX 4090".into(),
                kernel_driver: None,
                is_integrated: false,
                nvidia_gen: None,
            }],
        }
    }

    #[test]
    fn apply_writes_two_dropins_on_non_hybrid() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::Apply);
        let reports = apply(&ctx, &single_nvidia()).unwrap();
        assert_eq!(reports.len(), 3);
        assert!(dir
            .path()
            .join("etc/profile.d/99-nvidia-wayland.sh")
            .exists());
        assert!(dir
            .path()
            .join("etc/mkinitcpio.conf.d/nvidia-modules.conf")
            .exists());
    }

    #[test]
    fn apply_is_idempotent() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::Apply);
        apply(&ctx, &empty_gpus()).unwrap();
        let reports = apply(&ctx, &empty_gpus()).unwrap();
        for r in reports {
            assert!(matches!(r, ChangeReport::AlreadyApplied { .. }));
        }
    }

    #[test]
    fn dry_run_writes_nothing() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        apply(&ctx, &empty_gpus()).unwrap();
        assert!(!dir
            .path()
            .join("etc/profile.d/99-nvidia-wayland.sh")
            .exists());
    }
}
