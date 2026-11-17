//! A self-test for `scripts/check-private-strings.sh`, the gate behind public-sync's scrub tool
//! (procedure: `docs/ci/PUBLIC-SYNC.md`).
//!
//! It is the same set of cases, run against the REAL gate through [`std::process::Command`] rather
//! than re-derived in Rust — the point is to prove the shell script behaves, so re-implementing its
//! regex here would test the wrong thing. What the gate must get right, and what each case pins:
//!
//! * a commit about to go public may carry NO ledger trailer (`Lead:`, `Run:`, `Change:`, `Refs:`,
//!   `Fixes:`, `Lane-Run:`, `Written-By:`), yet the private `main` keeps them, so each is refused in
//!   `--public` mode and passes the default mode;
//! * a look-alike key — a real key with a suffix (`Change-Id:`), a different key sharing a prefix
//!   (`References:` vs `Refs:`), or a key not at the start of a line — is NOT refused;
//! * a clean message passes both modes;
//! * a private string is refused in BOTH modes, in the message and in a tracked file;
//! * misuse and an unset pattern are a hard stop (exit 2), never a silent pass.
//!
//! The gate reads its pattern from `LI_PRIVATE_PATTERNS`, falling back to `scripts/local.env`. This
//! test NEVER lets it reach that fallback: every invocation sets `LI_PRIVATE_PATTERNS` itself, to an
//! invented pattern of made-up strings — the "unset" case sets it to the EMPTY string, which the
//! gate treats as unset. So the real private patterns are never read, and the test runs on a host
//! with no `local.env` at all.
//!
//! The throwaway repositories are built under a [`tempfile::TempDir`] (which honours `TMPDIR`) that
//! is removed when it is dropped — including while a failing assertion unwinds. Host git config is
//! kept out of the fixture with `-c core.hooksPath` (this machine's commit hooks STAMP ledger
//! trailers, which would make the passing cases fail), `-c commit.gpgsign=false`, and an explicit
//! `user.name`/`user.email`.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The invented pattern the gate is handed — never the real `LI_PRIVATE_PATTERNS`.
const PATTERN: &str = "selftest-secret-host|/home/selftest-user";
/// One of the two strings `PATTERN` matches, used verbatim as the needle the gate must list.
const SECRET_HOST: &str = "selftest-secret-host";
const HOME_PATH: &str = "/home/selftest-user";

/// The ledger trailer keys `--public` mode refuses. Each is dropped by the scrub tool before a
/// commit goes public; the gate is the second line of defence if that ever stops happening.
const LEDGER_KEYS: [&str; 7] = [
    "Lead",
    "Run",
    "Change",
    "Refs",
    "Fixes",
    "Lane-Run",
    "Written-By",
];

/// A live fixture: the gate script, a throwaway git repo, and the temp dir that owns them (kept
/// only so its `Drop` cleans the whole thing up, even on panic).
struct Gate {
    _work: tempfile::TempDir,
    repo: PathBuf,
    nohooks: PathBuf,
    script: PathBuf,
}

/// One gate run: its exit code and the gate's stdout+stderr combined (findings go to stderr).
struct Run {
    code: i32,
    out: String,
}

impl Gate {
    /// Build a fixture, or `None` when the gate script is absent (a packaged/vendored build has no
    /// `scripts/`) — the same early-return the rest of the repo's tree-walking tests use.
    fn open() -> Option<Gate> {
        let script = gate_script();
        if !script.is_file() {
            eprintln!(
                "skip: {} is not present (not building from the repository)",
                script.display()
            );
            return None;
        }
        let work = tempfile::TempDir::new().expect("temp dir under $TMPDIR");
        let repo = work.path().join("repo");
        let nohooks = work.path().join("nohooks");
        std::fs::create_dir_all(&repo).expect("create fixture repo dir");
        std::fs::create_dir_all(&nohooks).expect("create empty hooks dir");
        let gate = Gate {
            _work: work,
            repo,
            nohooks,
            script,
        };
        gate.git(&["init", "-q", "-b", "main"]);
        // Uncommitted at first: the FIRST commit stages it, exactly as the shell self-test did.
        gate.write_file("notes.txt", "nothing private here\n");
        Some(gate)
    }

    /// `git -C <repo>` with the host's hooks, signing and identity kept out of the fixture.
    ///
    /// `GIT_CONFIG_COUNT=0` closes the OTHER route host config reaches a child git by — the
    /// `GIT_CONFIG_KEY_<n>`/`GIT_CONFIG_VALUE_<n>` environment a queue/lease harness injects to run
    /// `-c`-style config for its own jobs. Left as-is it both pollutes the fixture and breaks git
    /// outright the moment that injection is only half-present (a count with no matching key); zero
    /// makes the fixture read ONLY the `-c` overrides below, whatever the ambient environment is.
    fn git_cmd(&self) -> Command {
        let mut c = Command::new("git");
        c.env("GIT_CONFIG_COUNT", "0")
            .arg("-C")
            .arg(&self.repo)
            .arg("-c")
            .arg(format!("core.hooksPath={}", self.nohooks.display()))
            .arg("-c")
            .arg("commit.gpgsign=false")
            .arg("-c")
            .arg("user.name=selftest")
            .arg("-c")
            .arg("user.email=selftest@example.invalid");
        c
    }

    fn git(&self, args: &[&str]) {
        let out = self
            .git_cmd()
            .args(args)
            .output()
            .expect("run git against the fixture");
        assert!(
            out.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    fn write_file(&self, rel: &str, content: &str) {
        std::fs::write(self.repo.join(rel), content).expect("write fixture file");
    }

    /// Stage whatever the case wrote, commit with `message` (from stdin, so trailing newlines and a
    /// final `Ref`-style line survive verbatim), and print the new commit's id.
    fn commit(&self, message: &str) -> String {
        self.git(&["add", "-A"]);
        let mut child = self
            .git_cmd()
            .args(["commit", "-q", "--allow-empty", "-F", "-"])
            .stdin(Stdio::piped())
            .spawn()
            .expect("spawn git commit");
        child
            .stdin
            .take()
            .expect("commit stdin")
            .write_all(message.as_bytes())
            .expect("write the commit message");
        let status = child.wait().expect("wait for git commit");
        assert!(status.success(), "git commit failed (status {status})");

        let out = self
            .git_cmd()
            .args(["rev-parse", "HEAD"])
            .output()
            .expect("git rev-parse");
        assert!(out.status.success(), "git rev-parse HEAD failed");
        String::from_utf8(out.stdout)
            .expect("utf-8 commit id")
            .trim()
            .to_string()
    }

    /// Run the real gate with `args` and `pattern` as `LI_PRIVATE_PATTERNS`. `LI_PRIVATE_REPO` always
    /// points at the fixture, so the scan never touches this repository. The gate shells out to `git`
    /// itself, so it gets the same `GIT_CONFIG_COUNT=0` isolation the fixture's own `git` calls do.
    fn run(&self, args: &[&str], pattern: &str) -> Run {
        let out = Command::new("bash")
            .arg(&self.script)
            .args(args)
            .env("LI_PRIVATE_PATTERNS", pattern)
            .env("LI_PRIVATE_REPO", &self.repo)
            .env("GIT_CONFIG_COUNT", "0")
            .output()
            .expect("run scripts/check-private-strings.sh");
        let code = out.status.code().expect("gate exited on a signal");
        let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
        text.push_str(&String::from_utf8_lossy(&out.stderr));
        Run { code, out: text }
    }
}

/// The gate script, by absolute path from this crate's manifest.
fn gate_script() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("scripts")
        .join("check-private-strings.sh")
}

/// [1] Each ledger trailer key is refused in `--public` mode (listing the offending line) and
/// accepted in the default mode, where the private `main` still carries them.
#[test]
fn every_ledger_trailer_key_is_refused_only_in_public_mode() {
    let Some(gate) = Gate::open() else { return };
    for key in LEDGER_KEYS {
        let rev = gate.commit(&format!(
            "feat: add a thing\n\nA body paragraph, then the trailer.\n\n{key}: something\n"
        ));

        let dflt = gate.run(&[rev.as_str()], PATTERN);
        assert_eq!(
            dflt.code, 0,
            "default mode must still accept a `{key}` trailer; gate said:\n{}",
            dflt.out
        );

        let public = gate.run(&["--public", rev.as_str()], PATTERN);
        assert_eq!(
            public.code, 1,
            "--public must refuse a `{key}` trailer; gate said:\n{}",
            public.out
        );
        let needle = format!("{key}: something");
        assert!(
            public.out.contains(&needle),
            "--public must list the refused trailer {needle:?}; gate said:\n{}",
            public.out
        );
    }
}

/// [2] A key that is not at the start of a line, or is only the prefix of a longer key, is NOT a
/// ledger trailer — `--public` mode passes it.
#[test]
fn lookalike_keys_are_not_refused() {
    let Some(gate) = Gate::open() else { return };
    let rev = gate.commit(
        "docs: how the ledger works\n\nOnly write Lead: LI on a private commit, and\n\
         Change-Id: I123 is not Change: LI,\nReferences: LI-1 is not Refs: LI-1.\n",
    );

    assert_eq!(
        gate.run(&[rev.as_str()], PATTERN).code,
        0,
        "default mode must pass the near-miss message"
    );
    let public = gate.run(&["--public", rev.as_str()], PATTERN);
    assert_eq!(
        public.code, 0,
        "--public must NOT refuse `Lead` mid-line, `Change-Id:` or `References:`; gate said:\n{}",
        public.out
    );
}

/// [3] A clean message — no private string, no trailer — passes both modes.
#[test]
fn a_clean_message_passes_both_modes() {
    let Some(gate) = Gate::open() else { return };
    let rev =
        gate.commit("fix: keep the counters in step\n\nA plain message with no trailers at all.\n");

    assert_eq!(
        gate.run(&[rev.as_str()], PATTERN).code,
        0,
        "default mode must pass a clean message"
    );
    assert_eq!(
        gate.run(&["--public", rev.as_str()], PATTERN).code,
        0,
        "--public must pass a clean message"
    );
}

/// [3, cont.] A private string is refused in BOTH modes — in the message, and in a tracked file,
/// whether the gate scans a revision or the working tree (`--tree`).
#[test]
fn private_strings_are_refused_in_both_modes() {
    let Some(gate) = Gate::open() else { return };

    let rev = gate.commit(&format!(
        "chore: note the build box\n\n{SECRET_HOST} is where this was built.\n"
    ));
    for args in [&[rev.as_str()][..], &["--public", rev.as_str()][..]] {
        let run = gate.run(args, PATTERN);
        assert_eq!(
            run.code, 1,
            "a private string in the message must fail {args:?}; gate said:\n{}",
            run.out
        );
        assert!(
            run.out.contains(SECRET_HOST),
            "the gate must list the offending string; gate said:\n{}",
            run.out
        );
    }

    // The home path in a TRACKED file: the `<rev>` scan reads it, and `--tree` reads the working tree.
    gate.write_file(
        "notes.txt",
        &format!("build notes\n\nthe home path here is {HOME_PATH}\n"),
    );
    let rev = gate.commit("docs: record the build path\n");

    let scan = gate.run(&[rev.as_str()], PATTERN);
    assert_eq!(scan.code, 1, "a rev scan reads the tracked files too");
    assert!(
        scan.out.contains("notes.txt"),
        "the rev scan must list notes.txt; gate said:\n{}",
        scan.out
    );

    let tree = gate.run(&["--tree"], PATTERN);
    assert_eq!(tree.code, 1, "--tree reads the working tree");
    assert!(
        tree.out.contains("notes.txt"),
        "--tree must list notes.txt; gate said:\n{}",
        tree.out
    );
}

/// [4] Misuse and an unset pattern are a hard stop (exit 2), never a silent pass. The last case sets
/// `LI_PRIVATE_PATTERNS` to the EMPTY string — which the gate treats as unset — so it cannot fall
/// back to reading the real `local.env`.
#[test]
fn misuse_and_a_missing_pattern_exit_two() {
    let Some(gate) = Gate::open() else { return };

    assert_eq!(
        gate.run(&["--public"], PATTERN).code,
        2,
        "--public with no <rev> is a usage error"
    );
    assert_eq!(
        gate.run(&["--nosuchoption"], PATTERN).code,
        2,
        "an unknown option is a usage error"
    );
    assert_eq!(
        gate.run(&["--public", "--tree"], "").code,
        2,
        "an unset pattern is a hard stop, never a silent load of local.env"
    );
}
