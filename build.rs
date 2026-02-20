#[cfg(feature = "cuda")]
fn build_ptx() -> Result<()> {
    use cudaforge::{Error, KernelBuilder, Result};
    use std::env;
    use std::path::PathBuf;
    println!("cargo::rerun-if-changed=build.rs");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    let bindings = KernelBuilder::new()
        .source_dir("src") // Scan src/ for .cu files
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .build_ptx()?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn main() -> Result<()> {
    build_ptx()?;
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {}
