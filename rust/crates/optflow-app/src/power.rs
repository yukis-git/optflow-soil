//! Telling the operating system that a run is work the user is waiting for.
//!
//! A GUI app that is not in front, or whose window is covered, gets throttled, and a long analysis
//! is exactly when a user switches away. Measured on macOS (6240x4160 photos, window behind the
//! lock screen): about 30 s into a run App Nap moves every thread to priority 4, the efficiency
//! cores, and a frame takes up to four times as long. `UserInitiated` lifts that only to priority
//! 20, still about twice as long; `UserInteractive` (with latency critical) leaves the threads as
//! the command line tool's, which is not subject to any of this. On Windows the analogue is EcoQoS
//! for background processes (not observable on a CI runner).
//!
//! [`keep_running`] returns a guard; the exemption lasts while it lives. It also keeps the system
//! from going to sleep on idle (the display may still turn off).

/// While this lives the process is not throttled. Keep it on the thread that created it.
#[must_use = "the exemption ends when the guard is dropped"]
pub struct KeepRunning {
    #[cfg(target_os = "macos")]
    activity:
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2::runtime::NSObjectProtocol>>,
    /// Not `Send`: begin and end belong to one thread (and on Windows the sleep flag is per thread).
    _not_send: std::marker::PhantomData<*const ()>,
}

#[cfg(target_os = "macos")]
pub fn keep_running(reason: &str) -> KeepRunning {
    use objc2_foundation::{NSActivityOptions, NSProcessInfo, NSString};
    let activity = NSProcessInfo::processInfo().beginActivityWithOptions_reason(
        NSActivityOptions::UserInteractive,
        &NSString::from_str(reason),
    );
    KeepRunning {
        activity,
        _not_send: std::marker::PhantomData,
    }
}

#[cfg(target_os = "macos")]
impl Drop for KeepRunning {
    fn drop(&mut self) {
        // SAFETY: `activity` came from `beginActivityWithOptions:reason:` and is ended only here.
        unsafe { objc2_foundation::NSProcessInfo::processInfo().endActivity(&self.activity) };
    }
}

#[cfg(windows)]
mod win {
    use windows_sys::Win32::System::Power::{
        SetThreadExecutionState, ES_CONTINUOUS, ES_SYSTEM_REQUIRED,
    };
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, ProcessPowerThrottling, SetProcessInformation,
        PROCESS_POWER_THROTTLING_CURRENT_VERSION, PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
        PROCESS_POWER_THROTTLING_STATE,
    };

    /// `state_mask` 0 with the control bit set: never throttle; control mask 0: the system decides
    /// again. Failure only leaves the default behaviour, so it is not an error.
    fn set_throttling(control_mask: u32) -> bool {
        let state = PROCESS_POWER_THROTTLING_STATE {
            Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
            ControlMask: control_mask,
            StateMask: 0,
        };
        // SAFETY: the pointer and size describe `state`; the pseudo handle needs no closing.
        unsafe {
            SetProcessInformation(
                GetCurrentProcess(),
                ProcessPowerThrottling,
                std::ptr::from_ref(&state).cast(),
                std::mem::size_of_val(&state) as u32,
            ) != 0
        }
    }

    pub fn begin() -> bool {
        // SAFETY: no pointers; applies to the calling thread until `end`.
        unsafe { SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) };
        set_throttling(PROCESS_POWER_THROTTLING_EXECUTION_SPEED)
    }

    pub fn end() {
        set_throttling(0);
        // SAFETY: as in `begin`.
        unsafe { SetThreadExecutionState(ES_CONTINUOUS) };
    }
}

#[cfg(windows)]
pub fn keep_running(_reason: &str) -> KeepRunning {
    win::begin();
    KeepRunning {
        _not_send: std::marker::PhantomData,
    }
}

#[cfg(windows)]
impl Drop for KeepRunning {
    fn drop(&mut self) {
        win::end();
    }
}

#[cfg(not(any(target_os = "macos", windows)))]
pub fn keep_running(_reason: &str) -> KeepRunning {
    KeepRunning {
        _not_send: std::marker::PhantomData,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_guard_can_be_taken_and_ended_repeatedly_and_nested() {
        for _ in 0..3 {
            let a = keep_running("test");
            let b = keep_running("test nested");
            drop(a);
            drop(b);
        }
    }

    #[cfg(windows)]
    #[test]
    fn windows_accepts_the_throttling_call() {
        assert!(win::begin(), "SetProcessInformation failed");
        win::end();
    }
}
