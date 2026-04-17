use anyhow::{Context as _, Result};
use std::collections::HashSet;
use std::process::Command;

use crate::core::aur;
use crate::core::gpu::{GpuInventory, NvidiaGeneration, PackageSource};
use crate::core::state::TweakState;
use crate::core::{Context, ExecutionMode};
use crate::utils::fs_helper::{atomic_write, backup_to_dir, write_dropin, ChangeReport};
use crate::utils::process::run_streaming;

const SYSCTL_DROPIN_FILE: &str = "99-gaming.conf";
const SYSCTL_CONTENT: &str = "\
# Managed by arch-nvidia-tweaker — raises the mmap ceiling for modern games
# (Star Citizen, Hogwarts Legacy, Apex, etc.).
vm.max_map_count = 1048576
";

const ALWAYS_ON_GAMING_PACKAGES: &[&str] = &[
    "vulkan-icd-loader",
    "lib32-vulkan-icd-loader",
    "gamemode",
    "lib32-gamemode",
    "mangohud",
    "lib32-mangohud",
];

/// Gaming setup is Applied when: `[multilib]` is enabled AND every required repo package
/// for the detected GPU vendor(s) + the always-on set (vulkan-icd-loader / gamemode /
/// mangohud + their lib32 variants) is installed per `pacman -Qq`.
///
/// Never Incompatible — gaming setup is a universal improvement regardless of GPU vendor.
pub fn check_state(ctx: &Context, gpus: &GpuInventory) -> TweakState {
    if !is_multilib_enabled(&ctx.paths.pacman_conf) {
        return TweakState::Unapplied;
    }
    let expected = resolve_gaming_packages(gpus);
    let Some(installed) = pacman_query_installed_set(&expected) else {
        // pacman unavailable — can't verify; assume Unapplied so the button stays available.
        return TweakState::Unapplied;
    };
    if expected.iter().all(|p| installed.contains(p.as_str())) {
        TweakState::Applied
    } else {
        TweakState::Unapplied
    }
}

fn pacman_query_installed_set(names: &[String]) -> Option<HashSet<String>> {
    let out = Command::new("pacman").arg("-Qq").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let body = String::from_utf8_lossy(&out.stdout);
    let expected: HashSet<&str> = names.iter().map(String::as_str).collect();
    Some(
        body.lines()
            .filter(|l| expected.contains(l.trim()))
            .map(|l| l.trim().to_string())
            .collect(),
    )
}

pub fn apply(
    ctx: &Context,
    gpus: &GpuInventory,
    assume_yes: bool,
    progress: &mut dyn FnMut(&str),
) -> Result<Vec<ChangeReport>> {
    let mut reports = Vec::new();
    reports.push(enable_multilib(ctx)?);
    reports.push(write_sysctl_dropin(ctx)?);
    reports.push(install_official_packages(ctx, gpus, assume_yes, progress)?);

    let aur_pkgs = resolve_aur_packages(gpus);
    if !aur_pkgs.is_empty() {
        let refs: Vec<&str> = aur_pkgs.iter().map(String::as_str).collect();
        reports.push(aur::install_aur_packages(ctx, &refs, assume_yes, progress)?);
    }

    Ok(reports)
}

fn write_sysctl_dropin(ctx: &Context) -> Result<ChangeReport> {
    let target = ctx.paths.sysctl_d.join(SYSCTL_DROPIN_FILE);
    write_dropin(
        &target,
        SYSCTL_CONTENT,
        &ctx.paths.backup_dir,
        ctx.mode.is_dry_run(),
    )
}

fn enable_multilib(ctx: &Context) -> Result<ChangeReport> {
    let path = &ctx.paths.pacman_conf;
    let original =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;

    let (new_text, changed) = uncomment_multilib(&original);
    if !changed {
        return Ok(ChangeReport::AlreadyApplied {
            detail: format!("{}: [multilib] already enabled", path.display()),
        });
    }
    let detail = format!("{}: uncomment [multilib]", path.display());
    if ctx.mode.is_dry_run() {
        return Ok(ChangeReport::Planned { detail });
    }
    let backup = backup_to_dir(path, &ctx.paths.backup_dir)?;
    atomic_write(path, &new_text)?;
    Ok(ChangeReport::Applied { detail, backup })
}

fn install_official_packages(
    ctx: &Context,
    gpus: &GpuInventory,
    assume_yes: bool,
    progress: &mut dyn FnMut(&str),
) -> Result<ChangeReport> {
    let packages = resolve_gaming_packages(gpus);
    let detail = format!("pacman -Syu --needed {}", packages.join(" "));

    if ctx.mode.is_dry_run() {
        return Ok(ChangeReport::Planned { detail });
    }
    if matches!(ctx.mode, ExecutionMode::Apply) {
        let mut cmd = Command::new("pacman");
        cmd.arg("-Syu").arg("--needed");
        if assume_yes {
            cmd.arg("--noconfirm");
        }
        cmd.args(&packages);
        progress(&format!("[pacman] {detail}"));
        let status = run_streaming(cmd, |line| progress(&format!("[pacman] {line}")))?;
        if !status.success() {
            anyhow::bail!("pacman exited with {status}");
        }
    }
    Ok(ChangeReport::Applied {
        detail,
        backup: None,
    })
}

/// Official (repo) packages to install for gaming.
pub fn resolve_gaming_packages(gpus: &GpuInventory) -> Vec<String> {
    let mut pkgs: Vec<String> = ALWAYS_ON_GAMING_PACKAGES
        .iter()
        .map(|s| (*s).to_string())
        .collect();

    let mut add = |pkg: &str| {
        let s = pkg.to_string();
        if !pkgs.contains(&s) {
            pkgs.push(s);
        }
    };

    if let Some(nv) = gpus.primary_nvidia() {
        if let Some(rec) = nv.recommended_nvidia_package() {
            if rec.source == PackageSource::Official {
                add(rec.package);
            }
        }
        add("nvidia-utils");
        add("lib32-nvidia-utils");
        add("nvidia-settings");
        if gpus.is_hybrid() {
            add("nvidia-prime");
        }
    }
    if gpus.has_amd() {
        add("vulkan-radeon");
        add("lib32-vulkan-radeon");
    }
    if gpus.has_intel() {
        add("vulkan-intel");
        add("lib32-vulkan-intel");
    }

    pkgs
}

/// AUR packages the tool will build+install via yay. Empty for Turing+ hosts.
pub fn resolve_aur_packages(gpus: &GpuInventory) -> Vec<String> {
    let Some(nv) = gpus.primary_nvidia() else {
        return Vec::new();
    };
    let gen = nv.nvidia_gen.unwrap_or(NvidiaGeneration::Unknown);

    match gen {
        NvidiaGeneration::Maxwell | NvidiaGeneration::Kepler => vec![
            "nvidia-470xx-dkms".into(),
            "nvidia-470xx-utils".into(),
            "lib32-nvidia-470xx-utils".into(),
        ],
        NvidiaGeneration::Fermi => vec![
            "nvidia-390xx-dkms".into(),
            "nvidia-390xx-utils".into(),
            "lib32-nvidia-390xx-utils".into(),
        ],
        _ => Vec::new(),
    }
}

/// Read-only check: is the `[multilib]` repository already enabled in `pacman_conf`?
pub fn is_multilib_enabled<P: AsRef<std::path::Path>>(pacman_conf: P) -> bool {
    let Ok(body) = std::fs::read_to_string(pacman_conf.as_ref()) else {
        return false;
    };
    let mut in_multilib = false;
    for line in body.lines() {
        let t = line.trim();
        if t == "[multilib]" {
            in_multilib = true;
            continue;
        }
        if in_multilib && t.starts_with('[') {
            in_multilib = false;
        }
        if in_multilib && t.starts_with("Include") {
            return true;
        }
    }
    false
}

pub fn uncomment_multilib(original: &str) -> (String, bool) {
    enum State {
        Scanning,
        InMultilib,
        Done,
    }

    let mut out: Vec<String> = Vec::with_capacity(original.lines().count() + 1);
    let mut state = State::Scanning;
    let mut changed = false;

    for line in original.lines() {
        let t = line.trim();
        match state {
            State::Scanning => {
                if t == "#[multilib]" {
                    out.push("[multilib]".to_string());
                    state = State::InMultilib;
                    changed = true;
                } else if t == "[multilib]" {
                    out.push(line.to_string());
                    state = State::InMultilib;
                } else {
                    out.push(line.to_string());
                }
            }
            State::InMultilib => {
                if is_commented_include(t) {
                    out.push(uncomment_include(t));
                    state = State::Done;
                    changed = true;
                } else if is_uncommented_include(t) || t.starts_with('[') {
                    // Either we've hit the Include line (already uncommented) or the next
                    // section header — in both cases we're done scanning multilib.
                    out.push(line.to_string());
                    state = State::Done;
                } else {
                    out.push(line.to_string());
                }
            }
            State::Done => out.push(line.to_string()),
        }
    }

    let mut result = out.join("\n");
    if original.ends_with('\n') {
        result.push('\n');
    }
    (result, changed)
}

fn is_commented_include(line: &str) -> bool {
    let Some(rest) = line.strip_prefix('#') else {
        return false;
    };
    let rest = rest.trim_start();
    rest.starts_with("Include") && rest.contains('=')
}

fn is_uncommented_include(line: &str) -> bool {
    line.starts_with("Include") && line.contains('=')
}

fn uncomment_include(line: &str) -> String {
    line.trim_start_matches('#').trim_start().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::gpu::{GpuInfo, GpuVendor, NvidiaGeneration};

    fn inv(gpus: Vec<GpuInfo>) -> GpuInventory {
        GpuInventory { gpus }
    }

    fn nvidia(gen: NvidiaGeneration, device_id: u16) -> GpuInfo {
        GpuInfo {
            vendor: GpuVendor::Nvidia,
            vendor_id: 0x10de,
            device_id,
            pci_address: "0000:01:00.0".into(),
            vendor_name: "NVIDIA".into(),
            product_name: format!("dev {device_id:04x}"),
            kernel_driver: None,
            is_integrated: false,
            nvidia_gen: Some(gen),
        }
    }

    fn intel(device_id: u16) -> GpuInfo {
        GpuInfo {
            vendor: GpuVendor::Intel,
            vendor_id: 0x8086,
            device_id,
            pci_address: "0000:00:02.0".into(),
            vendor_name: "Intel".into(),
            product_name: format!("iGPU {device_id:04x}"),
            kernel_driver: None,
            is_integrated: true,
            nvidia_gen: None,
        }
    }

    fn amd(device_id: u16) -> GpuInfo {
        GpuInfo {
            vendor: GpuVendor::Amd,
            vendor_id: 0x1002,
            device_id,
            pci_address: "0000:03:00.0".into(),
            vendor_name: "AMD".into(),
            product_name: format!("Radeon {device_id:04x}"),
            kernel_driver: None,
            is_integrated: false,
            nvidia_gen: None,
        }
    }

    #[test]
    fn intel_only_gets_no_nvidia_packages() {
        let pkgs = resolve_gaming_packages(&inv(vec![intel(0x64a0)]));
        assert!(!pkgs.iter().any(|p| p.starts_with("nvidia")));
        assert!(pkgs.contains(&"vulkan-intel".to_string()));
    }

    #[test]
    fn nvidia_ada_desktop_gets_open_dkms_but_not_prime() {
        let pkgs = resolve_gaming_packages(&inv(vec![nvidia(NvidiaGeneration::Ada, 0x2684)]));
        assert!(pkgs.contains(&"nvidia-open-dkms".to_string()));
        assert!(!pkgs.iter().any(|p| p == "nvidia-prime"));
    }

    #[test]
    fn hybrid_gets_prime() {
        let pkgs = resolve_gaming_packages(&inv(vec![
            intel(0x3e9b),
            nvidia(NvidiaGeneration::Ampere, 0x25a2),
        ]));
        assert!(pkgs.contains(&"nvidia-prime".to_string()));
    }

    #[test]
    fn pascal_gets_legacy_nvidia_dkms_not_open() {
        let pkgs = resolve_gaming_packages(&inv(vec![nvidia(NvidiaGeneration::Pascal, 0x1B06)]));
        assert!(pkgs.contains(&"nvidia-dkms".to_string()));
        assert!(!pkgs.iter().any(|p| p == "nvidia-open-dkms"));
    }

    #[test]
    fn maxwell_official_list_excludes_aur_driver() {
        let pkgs = resolve_gaming_packages(&inv(vec![nvidia(NvidiaGeneration::Maxwell, 0x13C2)]));
        assert!(!pkgs.iter().any(|p| p.contains("470xx")));
    }

    #[test]
    fn maxwell_aur_list_contains_470xx_dkms_and_utils() {
        let aur = resolve_aur_packages(&inv(vec![nvidia(NvidiaGeneration::Maxwell, 0x13C2)]));
        assert!(aur.iter().any(|p| p == "nvidia-470xx-dkms"));
        assert!(aur.iter().any(|p| p == "nvidia-470xx-utils"));
        assert!(aur.iter().any(|p| p == "lib32-nvidia-470xx-utils"));
    }

    #[test]
    fn fermi_aur_list_is_390xx() {
        let aur = resolve_aur_packages(&inv(vec![nvidia(NvidiaGeneration::Fermi, 0x0DC4)]));
        assert!(aur.iter().any(|p| p == "nvidia-390xx-dkms"));
    }

    #[test]
    fn turing_or_newer_has_no_aur_requirements() {
        assert!(
            resolve_aur_packages(&inv(vec![nvidia(NvidiaGeneration::Turing, 0x1E04)])).is_empty()
        );
    }

    #[test]
    fn non_nvidia_never_requires_aur() {
        assert!(resolve_aur_packages(&inv(vec![intel(0x64a0)])).is_empty());
        assert!(resolve_aur_packages(&inv(vec![amd(0x73bf)])).is_empty());
    }

    const SAMPLE_COMMENTED: &str = "\
[core]
Include = /etc/pacman.d/mirrorlist

#[multilib]
#Include = /etc/pacman.d/mirrorlist
";

    const SAMPLE_ENABLED: &str = "\
[core]
Include = /etc/pacman.d/mirrorlist

[multilib]
Include = /etc/pacman.d/mirrorlist
";

    #[test]
    fn uncomments_multilib_section_and_include() {
        let (out, changed) = uncomment_multilib(SAMPLE_COMMENTED);
        assert!(changed);
        assert!(out.contains("\n[multilib]\n"));
    }

    #[test]
    fn idempotent_when_already_enabled() {
        let (_, changed) = uncomment_multilib(SAMPLE_ENABLED);
        assert!(!changed);
    }

    #[test]
    fn is_multilib_enabled_detects_active_section() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pacman.conf");
        std::fs::write(&path, SAMPLE_ENABLED).unwrap();
        assert!(is_multilib_enabled(&path));
    }

    #[test]
    fn is_multilib_enabled_detects_commented_section() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pacman.conf");
        std::fs::write(&path, SAMPLE_COMMENTED).unwrap();
        assert!(!is_multilib_enabled(&path));
    }
}
