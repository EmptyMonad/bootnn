fn main() {
    // Absolute path so the link script resolves regardless of the
    // directory cargo is invoked from (the harness builds from repo
    // root; a developer may build from rv/).
    println!(
        "cargo:rustc-link-arg=-T{}/link.ld",
        std::env::var("CARGO_MANIFEST_DIR").unwrap()
    );
    println!("cargo:rerun-if-changed=link.ld");
}
