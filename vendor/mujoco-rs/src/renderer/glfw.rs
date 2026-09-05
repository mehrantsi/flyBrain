use std::ffi::{CString, c_int, c_void};
use std::io;
use std::num::NonZero;
use std::ptr::NonNull;

use super::RendererError;

const GLFW_FALSE: c_int = 0;
const GLFW_VISIBLE: c_int = 0x0002_0004;

#[link(name = "glfw.3")]
unsafe extern "C" {
    fn glfwInit() -> c_int;
    fn glfwTerminate();
    fn glfwWindowHint(hint: c_int, value: c_int);
    fn glfwCreateWindow(
        width: c_int,
        height: c_int,
        title: *const i8,
        monitor: *mut c_void,
        share: *mut c_void,
    ) -> *mut c_void;
    fn glfwDestroyWindow(window: *mut c_void);
    fn glfwMakeContextCurrent(window: *mut c_void);
}

#[derive(Debug)]
pub(crate) struct GlStateGlfw {
    window: NonNull<c_void>,
}

impl GlStateGlfw {
    pub(crate) fn new(width: NonZero<u32>, height: NonZero<u32>) -> Result<Self, RendererError> {
        let width = c_int::try_from(width.get())
            .map_err(|_| RendererError::IoError(io::Error::other("GLFW width overflow")))?;
        let height = c_int::try_from(height.get())
            .map_err(|_| RendererError::IoError(io::Error::other("GLFW height overflow")))?;
        let title = CString::new("MuJoCo offscreen").unwrap();
        unsafe {
            if glfwInit() == GLFW_FALSE {
                return Err(RendererError::IoError(io::Error::other(
                    "GLFW initialization failed",
                )));
            }
            glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
            let window = NonNull::new(glfwCreateWindow(
                width,
                height,
                title.as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            ));
            let Some(window) = window else {
                glfwTerminate();
                return Err(RendererError::IoError(io::Error::other(
                    "GLFW context creation failed",
                )));
            };
            glfwMakeContextCurrent(window.as_ptr());
            Ok(Self { window })
        }
    }

    pub(crate) fn make_current(&self) -> Result<(), glutin::error::Error> {
        unsafe { glfwMakeContextCurrent(self.window.as_ptr()) };
        Ok(())
    }
}

impl Drop for GlStateGlfw {
    fn drop(&mut self) {
        unsafe {
            glfwMakeContextCurrent(std::ptr::null_mut());
            glfwDestroyWindow(self.window.as_ptr());
            glfwTerminate();
        }
    }
}
