use anyhow::{Context, Result};
use std::io::{BufRead, BufReader};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::thread;

/// Spawn `cmd` with stdin=/dev/null and both stdout/stderr piped line-by-line into `on_line`.
/// Merges stderr into the same stream (prefixed with `[err] `) so callers get a single log.
///
/// Returns the process `ExitStatus` once both streams close and the child has reaped. Callers
/// inspect `.success()` themselves.
///
/// Used for commands that can take tens of seconds to minutes (pacman, mkinitcpio, makepkg) —
/// cheap utilities should stay on plain `Command::status()`.
pub fn run_streaming<F>(mut cmd: Command, mut on_line: F) -> Result<ExitStatus>
where
    F: FnMut(&str),
{
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("spawning {:?}", cmd.get_program()))?;

    let stdout = child.stdout.take().expect("stdout piped");
    let stderr = child.stderr.take().expect("stderr piped");

    let (tx, rx) = mpsc::channel::<String>();

    let tx_out = tx.clone();
    let stdout_thread = thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if tx_out.send(line).is_err() {
                break;
            }
        }
    });

    let tx_err = tx.clone();
    let stderr_thread = thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            if tx_err.send(format!("[err] {line}")).is_err() {
                break;
            }
        }
    });

    drop(tx);

    while let Ok(line) = rx.recv() {
        on_line(&line);
    }

    let _ = stdout_thread.join();
    let _ = stderr_thread.join();
    child.wait().context("waiting on child process")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_captures_stdout_and_stderr() {
        let mut cmd = Command::new("sh");
        cmd.args([
            "-c",
            "echo out1; echo out2; echo err1 >&2; echo out3; echo err2 >&2",
        ]);
        let mut seen = Vec::<String>::new();
        let status = run_streaming(cmd, |l| seen.push(l.to_string())).unwrap();
        assert!(status.success());
        assert!(seen.iter().any(|l| l == "out1"));
        assert!(seen.iter().any(|l| l == "out2"));
        assert!(seen.iter().any(|l| l == "out3"));
        assert!(seen.iter().any(|l| l == "[err] err1"));
        assert!(seen.iter().any(|l| l == "[err] err2"));
    }

    #[test]
    fn streaming_propagates_nonzero_exit() {
        let mut cmd = Command::new("sh");
        cmd.args(["-c", "exit 7"]);
        let status = run_streaming(cmd, |_| {}).unwrap();
        assert!(!status.success());
        assert_eq!(status.code(), Some(7));
    }
}
