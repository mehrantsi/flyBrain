fn main() {}

#[cfg(target_os = "emscripten")]
#[used]
static KEEP_BROWSER_API: extern "C" fn(i32) -> i32 = flybrain_engine::browser::fb_init;
