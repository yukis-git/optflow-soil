//! Desktop app for optflow: the portable logic (settings, running a job, reading results,
//! statistics, export, colour maps, fonts, messages). `main.rs` is only the egui front end, so
//! everything here is unit-testable on any platform.

pub mod colormap;
pub mod crack;
pub mod export;
pub mod fields;
pub mod fonts;
pub mod job;
pub mod messages;
pub mod power;
pub mod resume;
pub mod roi_edit;
pub mod settings;
pub mod smoke;
pub mod source;
pub mod stats;
pub mod view;
