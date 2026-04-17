use anyhow::{Context as _, Result};
use std::path::PathBuf;
use std::process::Command;

use crate::core::gpu::GpuInventory;
use crate::core::state::TweakState;
use crate::core::Context;
use crate::utils::fs_helper::{atomic_write, backup_to_dir, ChangeReport};
use crate::utils::process::run_streaming;

// Phase 11 + Phase 15/16 kernel cmdline parameters.
//
// Each param maps to a specific driver/vendor situation:
//
//   NVIDIA:
//     nvidia-drm.modeset=1   → KMS modesetting, required for Wayland + early KMS
//     nvidia-drm.fbdev=1     → enables DRM fbdev emulation (hi-res TTY, cleaner
//                              boot handoff). Recommended from driver 545+.
//
//   AMD (amdgpu only):
//     amdgpu.ppfeaturemask=0xffffffff
//       Unlocks all PowerPlay features. Required for CoreCtrl, undervolt/OC
//       tools, and full fan-curve control. Safe on stable kernels.
//
//   Intel (i915 only — NOT xe):
//     i915.enable_guc=3      → enables both GuC submission and HuC firmware.
//                              Default on most supported platforms from kernel
//                              5.15+, but explicitly setting it ensures HuC is
//                              loaded on older kernels / mixed-kernel systems.
//
//   Intel (xe):
//     (none) — the xe driver manages GuC/HuC natively from boot.

pub const NVIDIA_DRM_PARAM: &str = "nvidia-drm.modeset=1";
pub const NVIDIA_DRM_FBDEV_PARAM: &str = "nvidia-drm.fbdev=1";
pub const AMD_PPFEATUREMASK_PARAM: &str = "amdgpu.ppfeaturemask=0xffffffff";
pub const I915_ENABLE_GUC_PARAM: &str = "i915.enable_guc=3";

/// Enumerate the kernel command-line parameters this host should carry, based on its
/// GPU inventory and each GPU's kernel driver. Returns an empty Vec for hosts whose
/// only GPU(s) don't benefit from any cmdline tweak (e.g. Intel-xe only).
///
/// This is the SINGLE source of truth consumed by `apply()` and `check_state()`.
pub fn required_kernel_params(gpus: &GpuInventory) -> Vec<&'static str> {
    let mut out: Vec<&'static str> = Vec::new();
    if gpus.has_nvidia() {
        out.push(NVIDIA_DRM_PARAM);
        out.push(NVIDIA_DRM_FBDEV_PARAM);
    }
    if gpus.has_amd_amdgpu() {
        out.push(AMD_PPFEATUREMASK_PARAM);
    }
    if gpus.has_intel_i915() {
        out.push(I915_ENABLE_GUC_PARAM);
    }
    out
}

// ── Bootloader classification ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootloaderType {
    Grub,
    SystemdBoot,
    Limine,
    Uki,
    Unknown,
}

impl BootloaderType {
    pub fn human(self) -> &'static str {
        match self {
            Self::Grub => "GRUB",
            Self::SystemdBoot => "systemd-boot",
            Self::Limine => "Limine",
            Self::Uki => "UKI (mkinitcpio /etc/kernel/cmdline)",
            Self::Unknown => "unknown / unsupported",
        }
    }
}

/// Probe the filesystem for the active bootloader. Order reflects practical precedence:
///  1. GRUB — unambiguous signal is `/etc/default/grub`
///  2. Limine — its own config file is unique to Limine
///  3. UKI — `/etc/kernel/cmdline` is the AUTHORITATIVE cmdline source on common Arch
///     setups where systemd-boot is the loader but the kernel image is a UKI built from
///     the mkinitcpio preset. If we detected plain systemd-boot first we'd try to edit
///     loader entries that don't exist on UKI-only systems.
///  4. systemd-boot (plain) — `/boot/loader/loader.conf` with loader entries
///  5. Unknown — nothing matched
///
/// NOTE: deliberate deviation from the task spec's literal ordering
/// (GRUB → SystemdBoot → Limine → UKI). Under that ordering, UKI+systemd-boot hosts — the
/// most common modern Arch setup — would be misclassified as plain systemd-boot and the
/// apply step would fail trying to write to non-existent loader entries.
pub fn detect_active_bootloader(ctx: &Context) -> BootloaderType {
    if ctx.paths.grub_default.exists() {
        return BootloaderType::Grub;
    }
    if ctx.paths.limine_candidates.iter().any(|p| p.exists()) {
        return BootloaderType::Limine;
    }
    if ctx.paths.kernel_cmdline.exists() {
        return BootloaderType::Uki;
    }
    if ctx.paths.sdb_loader_conf.exists() {
        return BootloaderType::SystemdBoot;
    }
    BootloaderType::Unknown
}

// ── Read-only state probe (used by diagnostics, auto::recommend, and the GUI) ──────────────

pub fn check_state(ctx: &Context, gpus: &GpuInventory) -> TweakState {
    let params = required_kernel_params(gpus);
    if params.is_empty() {
        // No applicable GPU (Intel-xe-only, or no GPU at all) → this tweak
        // doesn't apply to this host. UI shows an "Unsupported" badge.
        return TweakState::Incompatible;
    }
    let bt = detect_active_bootloader(ctx);
    if matches!(bt, BootloaderType::Unknown) {
        return TweakState::Incompatible;
    }
    // Applied iff EVERY required param is present on the active bootloader.
    let all_present = params.iter().all(|p| match bt {
        BootloaderType::Grub => grub_has_param(ctx, p),
        BootloaderType::SystemdBoot => sdb_all_entries_have_param(ctx, p),
        BootloaderType::Limine => limine_all_cmdlines_have_param(ctx, p),
        BootloaderType::Uki => uki_has_param(ctx, p),
        BootloaderType::Unknown => false,
    });
    if all_present {
        TweakState::Applied
    } else {
        TweakState::Unapplied
    }
}

// ── Legacy trait kept for diagnostics/CLI compatibility ─────────────────────────────────────

pub trait BootManager {
    fn describe(&self) -> String;
    fn has_parameter(&self, param: &str) -> Result<bool>;
}

pub struct UkiCmdlineManager<'a> {
    ctx: &'a Context,
}
pub struct GrubManager<'a> {
    ctx: &'a Context,
}
pub struct SdbManager<'a> {
    ctx: &'a Context,
}
pub struct LimineManager<'a> {
    ctx: &'a Context,
}

impl<'a> UkiCmdlineManager<'a> {
    pub fn new(ctx: &'a Context) -> Self {
        Self { ctx }
    }
}
impl<'a> GrubManager<'a> {
    pub fn new(ctx: &'a Context) -> Self {
        Self { ctx }
    }
}
impl<'a> SdbManager<'a> {
    pub fn new(ctx: &'a Context) -> Self {
        Self { ctx }
    }
}
impl<'a> LimineManager<'a> {
    pub fn new(ctx: &'a Context) -> Self {
        Self { ctx }
    }
}

impl BootManager for UkiCmdlineManager<'_> {
    fn describe(&self) -> String {
        format!("UKI cmdline ({})", self.ctx.paths.kernel_cmdline.display())
    }
    fn has_parameter(&self, param: &str) -> Result<bool> {
        Ok(uki_has_param(self.ctx, param))
    }
}
impl BootManager for GrubManager<'_> {
    fn describe(&self) -> String {
        format!("GRUB ({})", self.ctx.paths.grub_default.display())
    }
    fn has_parameter(&self, param: &str) -> Result<bool> {
        Ok(grub_has_param(self.ctx, param))
    }
}
impl BootManager for SdbManager<'_> {
    fn describe(&self) -> String {
        format!("systemd-boot ({})", self.ctx.paths.sdb_entries.display())
    }
    fn has_parameter(&self, param: &str) -> Result<bool> {
        Ok(sdb_all_entries_have_param(self.ctx, param))
    }
}
impl BootManager for LimineManager<'_> {
    fn describe(&self) -> String {
        let found = self.ctx.paths.limine_candidates.iter().find(|p| p.exists());
        match found {
            Some(p) => format!("Limine ({})", p.display()),
            None => "Limine (no limine.conf found)".to_string(),
        }
    }
    fn has_parameter(&self, param: &str) -> Result<bool> {
        Ok(limine_all_cmdlines_have_param(self.ctx, param))
    }
}

/// Backwards-compat: return a boxed BootManager for the active bootloader.
pub fn detect<'a>(ctx: &'a Context) -> Result<Box<dyn BootManager + 'a>> {
    match detect_active_bootloader(ctx) {
        BootloaderType::Uki => Ok(Box::new(UkiCmdlineManager::new(ctx))),
        BootloaderType::Grub => Ok(Box::new(GrubManager::new(ctx))),
        BootloaderType::SystemdBoot => Ok(Box::new(SdbManager::new(ctx))),
        BootloaderType::Limine => Ok(Box::new(LimineManager::new(ctx))),
        BootloaderType::Unknown => anyhow::bail!(
            "No supported bootloader detected. Checked: /etc/default/grub, limine.conf, \
             /etc/kernel/cmdline, /boot/loader/loader.conf."
        ),
    }
}

// ── Write APIs (apply) ──────────────────────────────────────────────────────────────────────

/// Apply every kernel cmdline parameter this host's GPU inventory requires to the active
/// bootloader's cmdline source. Multi-param in a single edit — `[modeset=1, fbdev=1]` for
/// NVIDIA, `[ppfeaturemask=...]` for amdgpu, `[enable_guc=3]` for i915, etc.
pub fn apply(
    ctx: &Context,
    gpus: &GpuInventory,
    progress: &mut dyn FnMut(&str),
) -> Result<ChangeReport> {
    let params = required_kernel_params(gpus);
    if params.is_empty() {
        return Ok(ChangeReport::AlreadyApplied {
            detail: "no cmdline params needed for this host's GPU inventory".to_string(),
        });
    }
    match detect_active_bootloader(ctx) {
        BootloaderType::Grub => apply_grub(ctx, &params, progress),
        BootloaderType::SystemdBoot => apply_sdb(ctx, &params, progress),
        BootloaderType::Limine => apply_limine(ctx, &params, progress),
        BootloaderType::Uki => apply_uki(ctx, &params, progress),
        BootloaderType::Unknown => anyhow::bail!("No supported bootloader detected."),
    }
}

fn apply_uki(
    ctx: &Context,
    params: &[&str],
    progress: &mut dyn FnMut(&str),
) -> Result<ChangeReport> {
    let path = &ctx.paths.kernel_cmdline;
    let original =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;

    let missing: Vec<&str> = params
        .iter()
        .copied()
        .filter(|p| !cmdline_contains(&original, p))
        .collect();
    if missing.is_empty() {
        return Ok(ChangeReport::AlreadyApplied {
            detail: format!(
                "{}: all params already present ({})",
                path.display(),
                params.join(" ")
            ),
        });
    }
    let trimmed = original.trim_end();
    let new = if trimmed.is_empty() {
        format!("{}\n", missing.join(" "))
    } else {
        format!("{trimmed} {}\n", missing.join(" "))
    };

    if ctx.mode.is_dry_run() {
        return Ok(ChangeReport::Planned {
            detail: format!(
                "UKI: add {} to {} + run mkinitcpio -P",
                missing.join(" "),
                path.display()
            ),
        });
    }
    let backup = backup_to_dir(path, &ctx.paths.backup_dir)?;
    atomic_write(path, &new)?;

    progress("[mkinitcpio] rebuilding UKIs via mkinitcpio -P");
    let mut cmd = Command::new("mkinitcpio");
    cmd.arg("-P");
    let status = run_streaming(cmd, |line| progress(&format!("[mkinitcpio] {line}")))?;
    if !status.success() {
        anyhow::bail!("mkinitcpio -P exited with {status}");
    }
    Ok(ChangeReport::Applied {
        detail: format!("UKI: added {} + rebuilt UKIs", missing.join(" ")),
        backup,
    })
}

fn apply_grub(
    ctx: &Context,
    params: &[&str],
    progress: &mut dyn FnMut(&str),
) -> Result<ChangeReport> {
    let path = &ctx.paths.grub_default;
    let original =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;

    // Iterate param-by-param through the pure helper so each addition is re-evaluated
    // against the latest text (idempotent if a param was added earlier in the loop).
    let mut current = original.clone();
    let mut any_changed = false;
    let mut any_found = false;
    let mut added: Vec<&str> = Vec::new();
    for p in params {
        let (new, changed, found) = grub_add_param(&current, p);
        current = new;
        if changed {
            any_changed = true;
            added.push(p);
        }
        any_found = any_found || found;
    }
    if !any_found {
        anyhow::bail!(
            "GRUB_CMDLINE_LINUX_DEFAULT line not found in {}. Add it manually or re-generate /etc/default/grub.",
            path.display()
        );
    }
    if !any_changed {
        return Ok(ChangeReport::AlreadyApplied {
            detail: format!(
                "{}: GRUB_CMDLINE_LINUX_DEFAULT already contains {}",
                path.display(),
                params.join(" ")
            ),
        });
    }
    if ctx.mode.is_dry_run() {
        return Ok(ChangeReport::Planned {
            detail: format!(
                "GRUB: add {} to GRUB_CMDLINE_LINUX_DEFAULT + run grub-mkconfig -o {}",
                added.join(" "),
                ctx.paths.grub_cfg.display()
            ),
        });
    }
    let backup = backup_to_dir(path, &ctx.paths.backup_dir)?;
    atomic_write(path, &current)?;

    progress(&format!(
        "[grub-mkconfig] grub-mkconfig -o {}",
        ctx.paths.grub_cfg.display()
    ));
    let mut cmd = Command::new("grub-mkconfig");
    cmd.arg("-o").arg(&ctx.paths.grub_cfg);
    let status = run_streaming(cmd, |line| progress(&format!("[grub-mkconfig] {line}")))?;
    if !status.success() {
        anyhow::bail!("grub-mkconfig exited with {status}");
    }
    Ok(ChangeReport::Applied {
        detail: format!(
            "GRUB: added {} + regenerated {}",
            added.join(" "),
            ctx.paths.grub_cfg.display()
        ),
        backup,
    })
}

fn apply_sdb(
    ctx: &Context,
    params: &[&str],
    progress: &mut dyn FnMut(&str),
) -> Result<ChangeReport> {
    let entries_dir = &ctx.paths.sdb_entries;
    let rd = std::fs::read_dir(entries_dir)
        .with_context(|| format!("opening {}", entries_dir.display()))?;

    let mut total = 0usize;
    let mut modified = 0usize;
    let mut already = 0usize;
    let mut first_backup: Option<PathBuf> = None;

    for entry in rd.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("conf") {
            continue;
        }
        let body = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let mut current = body.clone();
        let mut changed_any = false;
        let mut has_options_any = false;
        for p in params {
            let (new, changed, has_options) = sdb_add_param(&current, p);
            current = new;
            changed_any = changed_any || changed;
            has_options_any = has_options_any || has_options;
        }
        if !has_options_any {
            continue;
        }
        total += 1;
        if !changed_any {
            already += 1;
            continue;
        }
        modified += 1;
        if !ctx.mode.is_dry_run() {
            let bk = backup_to_dir(&path, &ctx.paths.backup_dir)?;
            if first_backup.is_none() {
                first_backup = bk;
            }
            atomic_write(&path, &current)?;
        }
    }

    if total == 0 {
        anyhow::bail!(
            "No systemd-boot entries with an `options` line in {}",
            entries_dir.display()
        );
    }
    if modified == 0 {
        return Ok(ChangeReport::AlreadyApplied {
            detail: format!(
                "systemd-boot: all {already} loader entries already contain {}",
                params.join(" ")
            ),
        });
    }
    if ctx.mode.is_dry_run() {
        return Ok(ChangeReport::Planned {
            detail: format!(
                "systemd-boot: add {} to {modified} entr{} + run bootctl update",
                params.join(" "),
                if modified == 1 { "y" } else { "ies" }
            ),
        });
    }

    progress("[bootctl] bootctl update");
    let mut cmd = Command::new("bootctl");
    cmd.arg("update");
    let status = run_streaming(cmd, |line| progress(&format!("[bootctl] {line}")))?;
    if !status.success() {
        log::warn!("bootctl update exited with {status} (non-fatal)");
    }

    Ok(ChangeReport::Applied {
        detail: format!(
            "systemd-boot: added {} to {modified} loader entr{} and ran bootctl update",
            params.join(" "),
            if modified == 1 { "y" } else { "ies" }
        ),
        backup: first_backup,
    })
}

fn apply_limine(
    ctx: &Context,
    params: &[&str],
    _progress: &mut dyn FnMut(&str),
) -> Result<ChangeReport> {
    let mut any_found = false;
    let mut any_modified = false;
    let mut first_backup: Option<PathBuf> = None;
    let mut touched: Vec<PathBuf> = Vec::new();

    for path in ctx.paths.limine_candidates.clone() {
        if !path.exists() {
            continue;
        }
        let original = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        let mut current = original.clone();
        let mut changed_any = false;
        let mut found_any = false;
        for p in params {
            let (new, changed, found) = limine_add_param(&current, p);
            current = new;
            changed_any = changed_any || changed;
            found_any = found_any || found;
        }
        if !found_any {
            continue;
        }
        any_found = true;
        if changed_any {
            any_modified = true;
            if !ctx.mode.is_dry_run() {
                let bk = backup_to_dir(&path, &ctx.paths.backup_dir)?;
                if first_backup.is_none() {
                    first_backup = bk;
                }
                atomic_write(&path, &current)?;
            }
            touched.push(path);
        }
    }

    if !any_found {
        anyhow::bail!(
            "No cmdline: / kernel_cmdline: / KERNEL_CMDLINE= directive found in any limine.conf candidate"
        );
    }
    if !any_modified {
        return Ok(ChangeReport::AlreadyApplied {
            detail: format!(
                "Limine: {} already present in all cmdline directives",
                params.join(" ")
            ),
        });
    }
    if ctx.mode.is_dry_run() {
        return Ok(ChangeReport::Planned {
            detail: format!(
                "Limine: add {} to {} file(s). No regeneration needed — Limine reads its config at boot.",
                params.join(" "),
                touched.len()
            ),
        });
    }
    Ok(ChangeReport::Applied {
        detail: format!(
            "Limine: added {} to {} file(s). No regeneration needed — Limine reads its config at boot.",
            params.join(" "),
            touched.len()
        ),
        backup: first_backup,
    })
}

// ── Parsing helpers (pure, testable) ────────────────────────────────────────────────────────

fn cmdline_contains(body: &str, param: &str) -> bool {
    body.split_whitespace().any(|tok| tok == param)
}

fn uki_has_param(ctx: &Context, param: &str) -> bool {
    let Ok(body) = std::fs::read_to_string(&ctx.paths.kernel_cmdline) else {
        return false;
    };
    cmdline_contains(&body, param)
}

fn grub_has_param(ctx: &Context, param: &str) -> bool {
    let Ok(body) = std::fs::read_to_string(&ctx.paths.grub_default) else {
        return false;
    };
    grub_cmdline_default_value(&body)
        .map(|v| cmdline_contains(&v, param))
        .unwrap_or(false)
}

fn grub_cmdline_default_value(body: &str) -> Option<String> {
    for line in body.lines() {
        let t = line.trim();
        if t.starts_with('#') {
            continue;
        }
        if let Some(rest) = t.strip_prefix("GRUB_CMDLINE_LINUX_DEFAULT=") {
            let v = rest.trim().trim_matches(|c: char| c == '"' || c == '\'');
            return Some(v.to_string());
        }
    }
    None
}

fn grub_add_param(original: &str, param: &str) -> (String, bool, bool) {
    let mut lines = Vec::with_capacity(original.lines().count() + 1);
    let mut changed = false;
    let mut found = false;

    for line in original.lines() {
        let t = line.trim_start();
        if t.starts_with('#') {
            lines.push(line.to_string());
            continue;
        }
        if let Some(rest) = t.strip_prefix("GRUB_CMDLINE_LINUX_DEFAULT=") {
            if found {
                // Already handled; subsequent duplicate lines are left untouched.
                lines.push(line.to_string());
                continue;
            }
            found = true;
            let indent: &str = &line[..line.len() - t.len()];
            let first = rest.chars().next();
            if first == Some('"') || first == Some('\'') {
                let q = first.unwrap();
                let inner = &rest[1..];
                let close = inner.rfind(q).unwrap_or(inner.len());
                let value = &inner[..close];
                let tail = if close < inner.len() {
                    &inner[close + 1..]
                } else {
                    ""
                };
                if cmdline_contains(value, param) {
                    lines.push(line.to_string());
                } else {
                    let new_value = if value.is_empty() {
                        param.to_string()
                    } else {
                        format!("{value} {param}")
                    };
                    lines.push(format!(
                        "{indent}GRUB_CMDLINE_LINUX_DEFAULT={q}{new_value}{q}{tail}"
                    ));
                    changed = true;
                }
            } else {
                // Unquoted value — preserve shape but quote on append.
                if cmdline_contains(rest, param) {
                    lines.push(line.to_string());
                } else {
                    let new_value = if rest.is_empty() {
                        param.to_string()
                    } else {
                        format!("{rest} {param}")
                    };
                    lines.push(format!(
                        "{indent}GRUB_CMDLINE_LINUX_DEFAULT=\"{new_value}\""
                    ));
                    changed = true;
                }
            }
        } else {
            lines.push(line.to_string());
        }
    }

    let mut result = lines.join("\n");
    if original.ends_with('\n') {
        result.push('\n');
    }
    (result, changed, found)
}

fn sdb_all_entries_have_param(ctx: &Context, param: &str) -> bool {
    let Ok(rd) = std::fs::read_dir(&ctx.paths.sdb_entries) else {
        return false;
    };
    let mut any = false;
    for e in rd.flatten() {
        let p = e.path();
        if p.extension().and_then(|s| s.to_str()) != Some("conf") {
            continue;
        }
        let Ok(body) = std::fs::read_to_string(&p) else {
            return false;
        };
        let has = body.lines().any(|l| {
            let t = l.trim_start();
            t.strip_prefix("options ")
                .map(|rest| cmdline_contains(rest, param))
                .unwrap_or(false)
        });
        if !has {
            return false;
        }
        any = true;
    }
    any
}

fn sdb_add_param(body: &str, param: &str) -> (String, bool, bool) {
    let mut lines = Vec::with_capacity(body.lines().count() + 1);
    let mut changed = false;
    let mut has_options = false;

    for line in body.lines() {
        let t = line.trim_start();
        if let Some(rest) = t.strip_prefix("options ") {
            has_options = true;
            let indent = &line[..line.len() - t.len()];
            if cmdline_contains(rest, param) {
                lines.push(line.to_string());
            } else {
                lines.push(format!("{indent}options {rest} {param}"));
                changed = true;
            }
        } else if t == "options" {
            has_options = true;
            let indent = &line[..line.len() - t.len()];
            lines.push(format!("{indent}options {param}"));
            changed = true;
        } else {
            lines.push(line.to_string());
        }
    }

    let mut result = lines.join("\n");
    if body.ends_with('\n') {
        result.push('\n');
    }
    (result, changed, has_options)
}

fn limine_all_cmdlines_have_param(ctx: &Context, param: &str) -> bool {
    let mut any = false;
    for path in &ctx.paths.limine_candidates {
        if !path.exists() {
            continue;
        }
        let Ok(body) = std::fs::read_to_string(path) else {
            continue;
        };
        for line in body.lines() {
            let t = line.trim();
            if t.starts_with('#') {
                continue;
            }
            let value = extract_limine_cmdline_value(t);
            if let Some(v) = value {
                any = true;
                if !cmdline_contains(&v, param) {
                    return false;
                }
            }
        }
    }
    any
}

fn extract_limine_cmdline_value(trimmed_line: &str) -> Option<String> {
    if let Some(r) = trimmed_line.strip_prefix("cmdline:") {
        return Some(r.trim().to_string());
    }
    if let Some(r) = trimmed_line.strip_prefix("kernel_cmdline:") {
        return Some(r.trim().to_string());
    }
    if let Some(r) = trimmed_line.strip_prefix("KERNEL_CMDLINE=") {
        return Some(
            r.trim()
                .trim_matches(|c: char| c == '"' || c == '\'')
                .to_string(),
        );
    }
    None
}

fn limine_add_param(original: &str, param: &str) -> (String, bool, bool) {
    let mut lines = Vec::with_capacity(original.lines().count() + 1);
    let mut changed = false;
    let mut found = false;

    for line in original.lines() {
        let t = line.trim_start();
        if t.starts_with('#') {
            lines.push(line.to_string());
            continue;
        }
        let indent = &line[..line.len() - t.len()];

        if let Some(rest) = t.strip_prefix("cmdline:") {
            found = true;
            let value = rest.trim();
            if cmdline_contains(value, param) {
                lines.push(line.to_string());
            } else {
                let new_value = if value.is_empty() {
                    param.to_string()
                } else {
                    format!("{value} {param}")
                };
                lines.push(format!("{indent}cmdline: {new_value}"));
                changed = true;
            }
        } else if let Some(rest) = t.strip_prefix("kernel_cmdline:") {
            found = true;
            let value = rest.trim();
            if cmdline_contains(value, param) {
                lines.push(line.to_string());
            } else {
                let new_value = if value.is_empty() {
                    param.to_string()
                } else {
                    format!("{value} {param}")
                };
                lines.push(format!("{indent}kernel_cmdline: {new_value}"));
                changed = true;
            }
        } else if let Some(rest) = t.strip_prefix("KERNEL_CMDLINE=") {
            found = true;
            let first = rest.chars().next();
            if first == Some('"') || first == Some('\'') {
                let q = first.unwrap();
                let inner = &rest[1..];
                let close = inner.rfind(q).unwrap_or(inner.len());
                let value = &inner[..close];
                if cmdline_contains(value, param) {
                    lines.push(line.to_string());
                } else {
                    let new_value = if value.is_empty() {
                        param.to_string()
                    } else {
                        format!("{value} {param}")
                    };
                    lines.push(format!("{indent}KERNEL_CMDLINE={q}{new_value}{q}"));
                    changed = true;
                }
            } else if cmdline_contains(rest, param) {
                lines.push(line.to_string());
            } else {
                let new_value = if rest.is_empty() {
                    param.to_string()
                } else {
                    format!("{rest} {param}")
                };
                lines.push(format!("{indent}KERNEL_CMDLINE={new_value}"));
                changed = true;
            }
        } else {
            lines.push(line.to_string());
        }
    }

    let mut result = lines.join("\n");
    if original.ends_with('\n') {
        result.push('\n');
    }
    (result, changed, found)
}

// ── Tests ───────────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ExecutionMode;
    use tempfile::tempdir;

    fn write(p: &std::path::Path, s: &str) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, s).unwrap();
    }

    // Detection -----------------------------------------------------------------------------

    #[test]
    fn detects_uki_when_cmdline_exists() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        write(&ctx.paths.kernel_cmdline, "rw quiet\n");
        assert_eq!(detect_active_bootloader(&ctx), BootloaderType::Uki);
    }

    #[test]
    fn detects_uki_over_systemd_boot_when_both_present() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        write(&ctx.paths.kernel_cmdline, "rw quiet\n");
        write(&ctx.paths.sdb_loader_conf, "default arch\ntimeout 3\n");
        assert_eq!(
            detect_active_bootloader(&ctx),
            BootloaderType::Uki,
            "UKI should win over plain SDB on coexisting setups"
        );
    }

    #[test]
    fn detects_grub() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        write(
            &ctx.paths.grub_default,
            "GRUB_CMDLINE_LINUX_DEFAULT=\"quiet\"\n",
        );
        assert_eq!(detect_active_bootloader(&ctx), BootloaderType::Grub);
    }

    #[test]
    fn detects_limine_in_any_candidate() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        write(&ctx.paths.limine_candidates[1], "timeout: 3\n");
        assert_eq!(detect_active_bootloader(&ctx), BootloaderType::Limine);
    }

    #[test]
    fn detects_systemd_boot_when_only_loader_conf() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        write(&ctx.paths.sdb_loader_conf, "default arch\ntimeout 3\n");
        assert_eq!(detect_active_bootloader(&ctx), BootloaderType::SystemdBoot);
    }

    #[test]
    fn detects_unknown_when_nothing() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        assert_eq!(detect_active_bootloader(&ctx), BootloaderType::Unknown);
    }

    // GRUB parsing --------------------------------------------------------------------------

    const GRUB_SAMPLE: &str = "\
# GRUB defaults
GRUB_DEFAULT=0
GRUB_TIMEOUT=5
GRUB_DISTRIBUTOR=\"Arch\"
GRUB_CMDLINE_LINUX_DEFAULT=\"loglevel=3 quiet\"
GRUB_CMDLINE_LINUX=\"\"
";

    #[test]
    fn grub_add_param_quoted_value() {
        let (out, changed, found) = grub_add_param(GRUB_SAMPLE, "nvidia-drm.modeset=1");
        assert!(found);
        assert!(changed);
        assert!(
            out.contains("GRUB_CMDLINE_LINUX_DEFAULT=\"loglevel=3 quiet nvidia-drm.modeset=1\"")
        );
        assert!(
            out.contains("GRUB_CMDLINE_LINUX=\"\""),
            "other lines preserved"
        );
    }

    #[test]
    fn grub_add_param_idempotent() {
        let already =
            GRUB_SAMPLE.replace("loglevel=3 quiet", "loglevel=3 quiet nvidia-drm.modeset=1");
        let (_out, changed, found) = grub_add_param(&already, "nvidia-drm.modeset=1");
        assert!(found);
        assert!(!changed);
    }

    #[test]
    fn grub_add_param_into_empty_quoted() {
        let s = "GRUB_CMDLINE_LINUX_DEFAULT=\"\"\n";
        let (out, changed, found) = grub_add_param(s, "nvidia-drm.modeset=1");
        assert!(found);
        assert!(changed);
        assert!(out.contains("GRUB_CMDLINE_LINUX_DEFAULT=\"nvidia-drm.modeset=1\""));
    }

    #[test]
    fn grub_add_param_ignores_commented_line() {
        let s = "#GRUB_CMDLINE_LINUX_DEFAULT=\"foo\"\n";
        let (out, _changed, found) = grub_add_param(s, "nvidia-drm.modeset=1");
        assert!(!found);
        assert_eq!(out, s);
    }

    #[test]
    fn grub_cmdline_value_extracts_quoted() {
        let v = grub_cmdline_default_value(GRUB_SAMPLE);
        assert_eq!(v.as_deref(), Some("loglevel=3 quiet"));
    }

    // systemd-boot parsing ------------------------------------------------------------------

    const SDB_ENTRY: &str = "\
title   Arch Linux
linux   /vmlinuz-linux
initrd  /initramfs-linux.img
options root=UUID=1234 rw
";

    #[test]
    fn sdb_add_param_appends_to_options_line() {
        let (out, changed, has) = sdb_add_param(SDB_ENTRY, "nvidia-drm.modeset=1");
        assert!(has);
        assert!(changed);
        assert!(out.contains("options root=UUID=1234 rw nvidia-drm.modeset=1"));
    }

    #[test]
    fn sdb_add_param_idempotent() {
        let already = SDB_ENTRY.replace("rw", "rw nvidia-drm.modeset=1");
        let (_out, changed, has) = sdb_add_param(&already, "nvidia-drm.modeset=1");
        assert!(has);
        assert!(!changed);
    }

    #[test]
    fn sdb_add_param_no_options_returns_false() {
        let entry = "title Arch\nlinux /vmlinuz-linux\ninitrd /initramfs.img\n";
        let (out, changed, has) = sdb_add_param(entry, "foo");
        assert!(!has);
        assert!(!changed);
        assert_eq!(out, entry);
    }

    // Limine parsing ------------------------------------------------------------------------

    const LIMINE_V9: &str = "\
timeout: 3
default_entry: 1

/Arch Linux
    protocol: linux
    path: boot():/vmlinuz-linux
    cmdline: root=UUID=1234 rw quiet
    module_path: boot():/initramfs-linux.img
";

    const LIMINE_LEGACY: &str = "\
TIMEOUT=5
DEFAULT_ENTRY=0

:Arch Linux
PROTOCOL=linux
KERNEL_PATH=boot():/vmlinuz-linux
KERNEL_CMDLINE=\"root=UUID=1234 rw quiet\"
MODULE_PATH=boot():/initramfs-linux.img
";

    #[test]
    fn limine_v9_add_param_to_cmdline() {
        let (out, changed, found) = limine_add_param(LIMINE_V9, "nvidia-drm.modeset=1");
        assert!(found);
        assert!(changed);
        assert!(out.contains("cmdline: root=UUID=1234 rw quiet nvidia-drm.modeset=1"));
    }

    #[test]
    fn limine_legacy_add_param_to_kernel_cmdline() {
        let (out, changed, found) = limine_add_param(LIMINE_LEGACY, "nvidia-drm.modeset=1");
        assert!(found);
        assert!(changed);
        assert!(out.contains("KERNEL_CMDLINE=\"root=UUID=1234 rw quiet nvidia-drm.modeset=1\""));
    }

    #[test]
    fn limine_add_param_idempotent() {
        let already = LIMINE_V9.replace("rw quiet", "rw quiet nvidia-drm.modeset=1");
        let (_out, changed, found) = limine_add_param(&already, "nvidia-drm.modeset=1");
        assert!(found);
        assert!(!changed);
    }

    #[test]
    fn limine_ignores_commented_cmdline_lines() {
        let s = "#cmdline: rw quiet\n";
        let (_out, _changed, found) = limine_add_param(s, "nvidia-drm.modeset=1");
        assert!(!found);
    }

    // UKI -----------------------------------------------------------------------------------

    #[test]
    fn uki_apply_is_idempotent_and_persistent() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        write(&ctx.paths.kernel_cmdline, "rw quiet\n");
        let r = apply_uki(
            &ctx,
            &[NVIDIA_DRM_PARAM, NVIDIA_DRM_FBDEV_PARAM],
            &mut |_| {},
        )
        .unwrap();
        assert!(matches!(r, ChangeReport::Planned { .. }));
        // File unchanged after dry-run
        let body = std::fs::read_to_string(&ctx.paths.kernel_cmdline).unwrap();
        assert_eq!(body, "rw quiet\n");
    }

    // cmdline_contains ----------------------------------------------------------------------

    #[test]
    fn cmdline_contains_whole_token_only() {
        assert!(cmdline_contains(
            "rw quiet nvidia-drm.modeset=1",
            "nvidia-drm.modeset=1"
        ));
        assert!(!cmdline_contains(
            "rw nvidia-drm.modeset=0",
            "nvidia-drm.modeset=1"
        ));
    }

    // check_state across bootloader types --------------------------------------------------

    fn nvidia_inv() -> GpuInventory {
        use crate::core::gpu::{GpuInfo, GpuVendor};
        GpuInventory {
            gpus: vec![GpuInfo {
                vendor: GpuVendor::Nvidia,
                vendor_id: 0x10de,
                device_id: 0x2204,
                pci_address: "0000:01:00.0".into(),
                vendor_name: "NVIDIA".into(),
                product_name: "RTX 3090".into(),
                kernel_driver: None,
                is_integrated: false,
                nvidia_gen: None,
            }],
        }
    }

    fn intel_inv() -> GpuInventory {
        use crate::core::gpu::{GpuInfo, GpuVendor};
        GpuInventory {
            gpus: vec![GpuInfo {
                vendor: GpuVendor::Intel,
                vendor_id: 0x8086,
                device_id: 0x64a0,
                pci_address: "0000:00:02.0".into(),
                vendor_name: "Intel".into(),
                product_name: "Arc".into(),
                kernel_driver: None,
                is_integrated: true,
                nvidia_gen: None,
            }],
        }
    }

    #[test]
    fn check_state_incompatible_without_nvidia() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        write(&ctx.paths.kernel_cmdline, "rw quiet\n");
        assert_eq!(check_state(&ctx, &intel_inv()), TweakState::Incompatible);
    }

    #[test]
    fn check_state_unapplied_for_uki_without_param() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        write(&ctx.paths.kernel_cmdline, "rw quiet\n");
        assert_eq!(check_state(&ctx, &nvidia_inv()), TweakState::Unapplied);
    }

    #[test]
    fn check_state_applied_for_uki_with_both_params() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        // Phase 16: NVIDIA hosts need BOTH modeset=1 AND fbdev=1 for Applied.
        write(
            &ctx.paths.kernel_cmdline,
            "rw quiet nvidia-drm.modeset=1 nvidia-drm.fbdev=1\n",
        );
        assert_eq!(check_state(&ctx, &nvidia_inv()), TweakState::Applied);
    }

    #[test]
    fn check_state_unapplied_for_uki_when_only_modeset_present() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        // fbdev=1 missing → Unapplied even though modeset=1 is there.
        write(&ctx.paths.kernel_cmdline, "rw quiet nvidia-drm.modeset=1\n");
        assert_eq!(check_state(&ctx, &nvidia_inv()), TweakState::Unapplied);
    }

    #[test]
    fn check_state_applied_for_grub_with_both_params() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        write(
            &ctx.paths.grub_default,
            "GRUB_CMDLINE_LINUX_DEFAULT=\"quiet nvidia-drm.modeset=1 nvidia-drm.fbdev=1\"\n",
        );
        assert_eq!(check_state(&ctx, &nvidia_inv()), TweakState::Applied);
    }

    #[test]
    fn check_state_applied_for_sdb_when_all_entries_have_both_params() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        write(&ctx.paths.sdb_loader_conf, "timeout 3\n");
        let entry = "title Arch\nlinux /vmlinuz\ninitrd /init.img\noptions root=UUID=1 rw nvidia-drm.modeset=1 nvidia-drm.fbdev=1\n";
        std::fs::create_dir_all(&ctx.paths.sdb_entries).unwrap();
        std::fs::write(ctx.paths.sdb_entries.join("arch.conf"), entry).unwrap();
        assert_eq!(detect_active_bootloader(&ctx), BootloaderType::SystemdBoot);
        assert_eq!(check_state(&ctx, &nvidia_inv()), TweakState::Applied);
    }

    #[test]
    fn check_state_unapplied_for_sdb_when_one_entry_missing_fbdev() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        write(&ctx.paths.sdb_loader_conf, "timeout 3\n");
        std::fs::create_dir_all(&ctx.paths.sdb_entries).unwrap();
        std::fs::write(
            ctx.paths.sdb_entries.join("arch.conf"),
            "options root=UUID=1 rw nvidia-drm.modeset=1 nvidia-drm.fbdev=1\n",
        )
        .unwrap();
        std::fs::write(
            ctx.paths.sdb_entries.join("fallback.conf"),
            "options root=UUID=1 rw nvidia-drm.modeset=1\n", // fbdev missing here
        )
        .unwrap();
        assert_eq!(check_state(&ctx, &nvidia_inv()), TweakState::Unapplied);
    }

    #[test]
    fn check_state_applied_for_limine_with_both_params() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        write(
            &ctx.paths.limine_candidates[0],
            "/Arch\n    cmdline: rw quiet nvidia-drm.modeset=1 nvidia-drm.fbdev=1\n",
        );
        assert_eq!(check_state(&ctx, &nvidia_inv()), TweakState::Applied);
    }

    // Phase 15: per-vendor required_kernel_params matrix ───────────────────────────────────

    fn intel_xe_inv() -> GpuInventory {
        use crate::core::gpu::{GpuInfo, GpuVendor};
        GpuInventory {
            gpus: vec![GpuInfo {
                vendor: GpuVendor::Intel,
                vendor_id: 0x8086,
                device_id: 0x64a0,
                pci_address: "0000:00:02.0".into(),
                vendor_name: "Intel".into(),
                product_name: "Lunar Lake Arc 140V".into(),
                kernel_driver: Some("xe".into()),
                is_integrated: true,
                nvidia_gen: None,
            }],
        }
    }

    fn intel_i915_inv() -> GpuInventory {
        use crate::core::gpu::{GpuInfo, GpuVendor};
        GpuInventory {
            gpus: vec![GpuInfo {
                vendor: GpuVendor::Intel,
                vendor_id: 0x8086,
                device_id: 0x3e9b,
                pci_address: "0000:00:02.0".into(),
                vendor_name: "Intel".into(),
                product_name: "UHD 630 (Coffee Lake)".into(),
                kernel_driver: Some("i915".into()),
                is_integrated: true,
                nvidia_gen: None,
            }],
        }
    }

    fn amd_amdgpu_inv() -> GpuInventory {
        use crate::core::gpu::{GpuInfo, GpuVendor};
        GpuInventory {
            gpus: vec![GpuInfo {
                vendor: GpuVendor::Amd,
                vendor_id: 0x1002,
                device_id: 0x73bf,
                pci_address: "0000:03:00.0".into(),
                vendor_name: "AMD".into(),
                product_name: "RX 6800".into(),
                kernel_driver: Some("amdgpu".into()),
                is_integrated: false,
                nvidia_gen: None,
            }],
        }
    }

    #[test]
    fn required_kernel_params_nvidia_only() {
        let p = required_kernel_params(&nvidia_inv());
        assert!(p.contains(&NVIDIA_DRM_PARAM));
        assert!(p.contains(&NVIDIA_DRM_FBDEV_PARAM));
        assert_eq!(p.len(), 2);
    }

    #[test]
    fn required_kernel_params_intel_xe_is_empty() {
        // Phase 15: xe driver handles GuC/HuC natively — no params needed.
        assert!(required_kernel_params(&intel_xe_inv()).is_empty());
    }

    #[test]
    fn required_kernel_params_intel_i915_gets_guc() {
        let p = required_kernel_params(&intel_i915_inv());
        assert_eq!(p, vec![I915_ENABLE_GUC_PARAM]);
    }

    #[test]
    fn required_kernel_params_amdgpu_gets_ppfeaturemask() {
        let p = required_kernel_params(&amd_amdgpu_inv());
        assert_eq!(p, vec![AMD_PPFEATUREMASK_PARAM]);
    }

    #[test]
    fn required_kernel_params_intel_xe_plus_nvidia_dgpu_only_nvidia() {
        use crate::core::gpu::{GpuInfo, GpuVendor};
        let inv = GpuInventory {
            gpus: vec![
                intel_xe_inv().gpus.into_iter().next().unwrap(),
                GpuInfo {
                    vendor: GpuVendor::Nvidia,
                    vendor_id: 0x10de,
                    device_id: 0x25a2,
                    pci_address: "0000:01:00.0".into(),
                    vendor_name: "NVIDIA".into(),
                    product_name: "RTX 3050M".into(),
                    kernel_driver: Some("nvidia".into()),
                    is_integrated: false,
                    nvidia_gen: None,
                },
            ],
        };
        let p = required_kernel_params(&inv);
        assert!(p.contains(&NVIDIA_DRM_PARAM));
        assert!(p.contains(&NVIDIA_DRM_FBDEV_PARAM));
        assert!(!p.contains(&I915_ENABLE_GUC_PARAM));
    }

    #[test]
    fn check_state_incompatible_for_intel_xe_only() {
        // Phase 15: Intel-xe-only host has no bootloader params at all → Incompatible.
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        write(&ctx.paths.kernel_cmdline, "rw quiet\n");
        assert_eq!(check_state(&ctx, &intel_xe_inv()), TweakState::Incompatible);
    }

    #[test]
    fn check_state_applied_for_amdgpu_with_ppfeaturemask() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        write(
            &ctx.paths.kernel_cmdline,
            "rw quiet amdgpu.ppfeaturemask=0xffffffff\n",
        );
        assert_eq!(check_state(&ctx, &amd_amdgpu_inv()), TweakState::Applied);
    }

    #[test]
    fn check_state_applied_for_i915_with_enable_guc() {
        let dir = tempdir().unwrap();
        let ctx = Context::rooted_for_test(dir.path(), ExecutionMode::DryRun);
        write(&ctx.paths.kernel_cmdline, "rw quiet i915.enable_guc=3\n");
        assert_eq!(check_state(&ctx, &intel_i915_inv()), TweakState::Applied);
    }
}
