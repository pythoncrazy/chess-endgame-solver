//! Records the source revision (with a +dirty mark) for table provenance.
use std::path::Path;
use std::process::Command;

fn main() {
    let git = |args: &[&str]| {
        Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    let revision = git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let dirty =
        git(&["status", "--porcelain", "--untracked-files=no"]).is_none_or(|s| !s.is_empty());
    let suffix = if dirty { "+dirty" } else { "" };
    println!("cargo:rustc-env=TB_SOURCE_REVISION={revision}{suffix}");
    // The mark is only correct if a change to any source runs this script again.
    for path in ["src", "Cargo.toml", "../src", "../Cargo.toml"] {
        println!("cargo:rerun-if-changed={path}");
    }
    if let Some(dir) = git(&["rev-parse", "--absolute-git-dir"]) {
        let head = git(&["symbolic-ref", "-q", "HEAD"]);
        for name in ["HEAD", "index", "packed-refs"]
            .into_iter()
            .chain(head.as_deref())
        {
            let path = Path::new(&dir).join(name);
            // Cargo runs the script on each build if a path does not exist.
            if path.exists() {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
}
