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
    /// Portable/laptop family: 8 Portable, 9 Laptop, 10 Notebook, 11 Hand Held, 14 Sub Notebook.
    /// Desktop family: 3 Desktop, 4 Low Profile Desktop, 6 Mini Tower, 7 Tower.
    pub fn from_chassis_type(code: u8) -> Self {
        match code {
            8 | 9 | 10 | 11 | 14 => Self::Laptop,
            3 | 4 | 6 | 7 => Self::Desktop,
            _ => Self::Unknown,
        }
    }
}

/// Read and parse the SMBIOS chassis_type sysfs file to determine the host's form factor.
pub fn detect<P: AsRef<Path>>(chassis_type_path: P) -> Result<FormFactor> {
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
        for code in [8u8, 9, 10, 11, 14] {
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
        for code in [0u8, 1, 2, 5, 12, 13, 15, 30, 99] {
            assert_eq!(
                FormFactor::from_chassis_type(code),
                FormFactor::Unknown,
                "code {code}"
            );
        }
    }
}
