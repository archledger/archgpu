// ═══════════════════════════════════════════════════════════════════════════════════════════
// Phase 21: CPU CET-IBT probe
// ═══════════════════════════════════════════════════════════════════════════════════════════
//
// Narrow-scope CPU introspection used by the bootloader tweak. The only signal we care
// about right now is whether the running CPU exposes Intel CET Indirect Branch Tracking
// (the `ibt` flag in /proc/cpuinfo). CET-IBT is present on Alder Lake (12th-gen Intel)
// and later, plus Zen 4+ AMD parts — and older NVIDIA drivers (pre-545) crash during
// module init on these CPUs because their indirect calls don't respect ENDBR landing
// pads. The Arch Wiki workaround is the `ibt=off` kernel cmdline parameter.
//
// We surface this as a BOOLEAN, not a CPU-name-to-generation mapping, because:
//   (a) /proc/cpuinfo's "model name" field is unreliable for generation detection
//       — linutil's `cut -c 2-3` ends up comparing strings like "AM" or "In" against
//       integers, which fails on every host. Your field report showed exactly this:
//       "[: AM: integer expected" on your Ryzen 5700G.
//   (b) the `ibt` flag is the ACTUAL property the kernel param gates on — if the CPU
//       has IBT the kernel enables the enforcement, and the NVIDIA driver (if old)
//       breaks. No need to infer from model names what the kernel is already telling us.
//
// Adding more CPU probes later (SEV-SNP, SME, x86-64-v4 selectors for CachyOS-style repo
// picking, etc.) should follow the same pattern — a pure `*_from_cpuinfo(text)` parser
// plus a thin runtime wrapper that reads the path from SystemPaths.
// ═══════════════════════════════════════════════════════════════════════════════════════════

use std::path::Path;

/// Pure: parse a /proc/cpuinfo-formatted string and return true iff any CPU line lists
/// the `ibt` flag. CET-IBT is a uniform extension across physical cores on a given CPU,
/// but hybrid architectures (Alder Lake's P-cores vs E-cores) can have per-core flag
/// differences — scan every flags line to be robust.
pub fn cpu_has_ibt_from_cpuinfo(cpuinfo: &str) -> bool {
    for line in cpuinfo.lines() {
        let trimmed = line.trim_start();
        // x86 uses "flags", ARM uses "Features" — scan both; /proc/cpuinfo may differ
        // across platforms but IBT is x86-only, so the ARM branch is there for
        // defensive robustness, not feature parity.
        if !(trimmed.starts_with("flags") || trimmed.starts_with("Features")) {
            continue;
        }
        let Some((_, rest)) = trimmed.split_once(':') else {
            continue;
        };
        for flag in rest.split_whitespace() {
            if flag == "ibt" {
                return true;
            }
        }
    }
    false
}

/// Runtime: read `/proc/cpuinfo` (or the test-rooted equivalent from SystemPaths.cpuinfo)
/// and delegate to the pure parser. Returns false on read error — conservative default
/// that means "don't add ibt=off" when we can't prove IBT is present.
pub fn cpu_has_ibt<P: AsRef<Path>>(cpuinfo_path: P) -> bool {
    let Ok(body) = std::fs::read_to_string(cpuinfo_path.as_ref()) else {
        return false;
    };
    cpu_has_ibt_from_cpuinfo(&body)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Real /proc/cpuinfo excerpt from an Alder Lake i7-12700K (IBT present).
    const ALDER_LAKE_CPUINFO: &str = "\
processor\t: 0
vendor_id\t: GenuineIntel
cpu family\t: 6
model\t\t: 151
model name\t: 12th Gen Intel(R) Core(TM) i7-12700K
stepping\t: 2
microcode\t: 0x2c
flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm ibt shstk pku ospke waitpkg gfni vaes vpclmulqdq
";

    // Real excerpt from Ryzen 7 5700G (Zen 3 / Cezanne) — predates CET-IBT in AMD
    // parts (which landed in Zen 4). Regression guard: the field-test host must NOT
    // trigger the `ibt=off` param.
    const RYZEN_5700G_CPUINFO: &str = "\
processor\t: 0
vendor_id\t: AuthenticAMD
cpu family\t: 25
model\t\t: 80
model name\t: AMD Ryzen 7 5700G with Radeon Graphics
stepping\t: 0
microcode\t: 0xa50000c
flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 invpcid_single hw_pstate ssbd mba ibrs ibpb stibp vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk clzero irperf xsaveerptr rdpru wbnoinvd cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold v_vmsave_vmload vgif x2avic v_spec_ctrl vnmi avic umip pku ospke vaes vpclmulqdq rdpid overflow_recov succor smca fsrm debug_swap
";

    // Minimal hand-crafted /proc/cpuinfo with no flags line — defensive test.
    const NO_FLAGS_LINE: &str = "\
processor\t: 0
vendor_id\t: GenuineIntel
";

    #[test]
    fn detects_ibt_on_alder_lake() {
        assert!(cpu_has_ibt_from_cpuinfo(ALDER_LAKE_CPUINFO));
    }

    #[test]
    fn does_not_detect_ibt_on_zen3_ryzen() {
        // Phase 21 regression guard: field-test host (Ryzen 5700G, Zen 3) must NOT
        // get `ibt=off`. Zen 4+ (7000 series, 2022+) is when AMD added CET-IBT.
        assert!(
            !cpu_has_ibt_from_cpuinfo(RYZEN_5700G_CPUINFO),
            "Ryzen 5700G (Zen 3) must not be detected as having IBT"
        );
    }

    #[test]
    fn safe_on_cpuinfo_without_flags_line() {
        assert!(!cpu_has_ibt_from_cpuinfo(NO_FLAGS_LINE));
    }

    #[test]
    fn does_not_match_ibt_as_substring_of_other_flags() {
        // Guard against a buggy implementation that matches "ibt" inside "ibtoff" or
        // "xibtx" — `split_whitespace` + exact equality rules this out, but test it
        // explicitly so a future refactor can't silently regress.
        let fake = "processor\t: 0\nflags\t\t: fpu vme xibtx xibt xibtx\n";
        assert!(!cpu_has_ibt_from_cpuinfo(fake));
    }

    #[test]
    fn detects_ibt_per_core_hybrid_architecture() {
        // Alder Lake P-cores support CET-IBT; E-cores (Gracemont) report the flag too
        // in current kernels, but older Linux revisions masked it. If ANY core has
        // the flag, the workaround is warranted.
        let hybrid = "\
processor\t: 0
flags\t\t: fpu vme de pse sse ibt
processor\t: 8
flags\t\t: fpu vme de pse sse
";
        assert!(cpu_has_ibt_from_cpuinfo(hybrid));
    }

    #[test]
    fn runtime_reads_cpuinfo_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cpuinfo");
        std::fs::write(&path, ALDER_LAKE_CPUINFO).unwrap();
        assert!(cpu_has_ibt(&path));

        std::fs::write(&path, RYZEN_5700G_CPUINFO).unwrap();
        assert!(!cpu_has_ibt(&path));
    }

    #[test]
    fn runtime_returns_false_for_missing_cpuinfo() {
        assert!(!cpu_has_ibt("/nonexistent/path/to/cpuinfo"));
    }
}
