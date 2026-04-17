fn main() {
    slint_build::compile("ui/main_window.slint").expect("Slint build failed");
    println!("cargo:rerun-if-changed=ui/main_window.slint");
}
