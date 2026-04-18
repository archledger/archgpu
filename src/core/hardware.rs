use anyhow::{Context as _, Result};
use std::path::Path;

/// Chassis form factor as interpreted from SMBIOS `chassis_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormFactor {
    Laptop,
    Desktop,
    Unknown,
}

impl FormFactor {
    /// Map SMBIOS chassis-type integers to coarse form factor.
    ///
    /// Portable/laptop family: 8 Portable, 9 Laptop, 10 Notebook, 11 Hand Held, 14 Sub Notebook,
    /// **31 Tablet, 32 Convertible** (Phase 19 — 2-in-1s and tablets run hybrid graphics too and
    /// need the same Optimus/power tweaks as traditional laptops).
    /// Desktop family: 3 Desktop, 4 Low Profile Desktop, 6 Mini Tower, 7 Tower.
    ///
    /// NOTE on Phase 19's multi-GPU desktop rule: `FormFactor::Desktop` combined with a
    /// hybrid GPU inventory means "user has both iGPU and dGPU, but on a tower the physical
    /// display cable dictates which GPU drives the monitor — PRIME offload is a laptop
    /// concept." Callers in `core::prime` and `core::gaming` check form before recommending
    /// `nvidia-prime`, writing the PRIME Xorg OutputClass drop-in, or setting
    /// `__NV_PRIME_RENDER_OFFLOAD=1`.
    pub fn from_chassis_type(code: u8) -> Self {
        match code {
            8 | 9 | 10 | 11 | 14 | 31 | 32 => Self::Laptop,
            3 | 4 | 6 | 7 => Self::Desktop,
            _ => Self::Unknown,
        }
    }
}

/// Read and parse the SMBIOS chassis_type sysfs file to determine the host's form factor.
///
/// Canonical name for the sysfs accessor — reads `/sys/class/dmi/id/chassis_type` (or the
/// test-rooted equivalent) and classifies the host as Laptop / Desktop / Unknown.
pub fn get_chassis_type<P: AsRef<Path>>(chassis_type_path: P) -> Result<FormFactor> {
    let path = chassis_type_path.as_ref();
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("reading chassis_type from {}", path.display()))?;
    let trimmed = raw.trim();
    let code: u8 = trimmed
        .parse()
        .with_context(|| format!("parsing chassis_type value {trimmed:?}"))?;
    Ok(FormFactor::from_chassis_type(code))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_laptop_codes() {
        // Phase 19: adds 31 (Tablet) and 32 (Convertible) to the laptop family.
        for code in [8u8, 9, 10, 11, 14, 31, 32] {
            assert_eq!(
                FormFactor::from_chassis_type(code),
                FormFactor::Laptop,
                "code {code}"
            );
        }
    }

    #[test]
    fn known_desktop_codes() {
        for code in [3u8, 4, 6, 7] {
            assert_eq!(
                FormFactor::from_chassis_type(code),
                FormFactor::Desktop,
                "code {code}"
            );
        }
    }

    #[test]
    fn unknown_codes() {
        // Codes 31, 32 are now Laptop (Phase 19) — no longer in the Unknown bucket.
        for code in [0u8, 1, 2, 5, 12, 13, 15, 30, 33, 99] {
            assert_eq!(
                FormFactor::from_chassis_type(code),
                FormFactor::Unknown,
                "code {code}"
            );
        }
    }

    #[test]
    fn get_chassis_type_reads_sysfs_mock() {
        // Phase 19: test the sysfs reader with a mocked chassis_type file. Covers the full
        // "read → parse → classify" pipeline the production path runs from
        // /sys/class/dmi/id/chassis_type.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("chassis_type");
        std::fs::write(&path, "10\n").unwrap();
        assert_eq!(get_chassis_type(&path).unwrap(), FormFactor::Laptop);

        std::fs::write(&path, "3\n").unwrap();
        assert_eq!(get_chassis_type(&path).unwrap(), FormFactor::Desktop);

        std::fs::write(&path, "32\n").unwrap();
        assert_eq!(
            get_chassis_type(&path).unwrap(),
            FormFactor::Laptop,
            "Convertible (code 32) must map to Laptop per Phase 19"
        );

        std::fs::write(&path, "99\n").unwrap();
        assert_eq!(get_chassis_type(&path).unwrap(), FormFactor::Unknown);
    }

    #[test]
    fn get_chassis_type_errors_on_malformed_sysfs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("chassis_type");
        std::fs::write(&path, "not-a-number\n").unwrap();
        assert!(get_chassis_type(&path).is_err());
    }
}
