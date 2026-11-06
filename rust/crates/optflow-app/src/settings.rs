//! The settings the user edits and how they become `--set key=value` overrides of the CLI.

/// A named set of values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    /// The strongest known configuration (hybrid sequences, temporal average of 3 frames each
    /// side, frozen pre-warp, reference switch at confidence 0.3).
    Recommended,
    /// `Config::default()` (the plain chain mode).
    Standard,
}

impl Preset {
    pub fn label(self) -> &'static str {
        match self {
            Preset::Recommended => "推奨",
            Preset::Standard => "標準",
        }
    }
}

/// The values the advanced panel edits. Anything not listed here keeps the `Config` default.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub seq_mode: String,
    pub tavg: i64,
    pub prewarp_freeze_at: i64,
    pub ref_update_conf: f64,
    pub max_disp: f64,
    pub conf_min: f64,
    pub sigma_w: f64,
    pub scale: i64,
    /// Crack model `crack_model = "xdic"` (experimental; not part of any preset).
    pub crack: bool,
}

impl Settings {
    pub fn preset(p: Preset) -> Settings {
        let d = optflow_core::config::Config::default();
        let base = Settings {
            seq_mode: d.seq_mode.clone(),
            tavg: d.tavg,
            prewarp_freeze_at: d.prewarp_freeze_at,
            ref_update_conf: d.ref_update_conf,
            max_disp: d.max_disp,
            conf_min: d.conf_min,
            sigma_w: d.sigma_w,
            scale: 1,
            crack: false,
        };
        match p {
            Preset::Standard => base,
            Preset::Recommended => Settings {
                seq_mode: "hybrid".into(),
                tavg: 3,
                prewarp_freeze_at: 3,
                ref_update_conf: 0.3,
                ..base
            },
        }
    }

    /// The preset these values equal, if any.
    pub fn matching_preset(&self) -> Option<Preset> {
        // the crack toggle is independent of the presets
        let plain = Settings {
            crack: false,
            ..self.clone()
        };
        [Preset::Recommended, Preset::Standard]
            .into_iter()
            .find(|p| plain == Settings::preset(*p))
    }

    /// The `--set key=value` overrides (every value of this struct is passed, so the run does not
    /// depend on defaults changing). `scale` is a separate option of the CLI, not a key.
    pub fn overrides(&self) -> Vec<String> {
        let mut v = vec![
            format!("seq_mode={}", self.seq_mode),
            format!("tavg={}", self.tavg),
            format!("prewarp_freeze_at={}", self.prewarp_freeze_at),
            format!("ref_update_conf={}", fmt_f(self.ref_update_conf)),
            format!("max_disp={}", fmt_f(self.max_disp)),
            format!("conf_min={}", fmt_f(self.conf_min)),
            format!("sigma_w={}", fmt_f(self.sigma_w)),
        ];
        if self.crack {
            v.push("crack_model=xdic".to_string());
        }
        v
    }
}

/// A float in the form the CLI parses back exactly (`Display` of f64 is the shortest round trip).
fn fmt_f(x: f64) -> String {
    format!("{x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use optflow_core::config::Config;

    fn config_of(s: &Settings) -> Config {
        optflow_cli::settings::config(None, &s.overrides()).unwrap()
    }

    #[test]
    fn recommended_is_the_strongest_known_config() {
        let c = config_of(&Settings::preset(Preset::Recommended));
        assert_eq!(c.seq_mode, "hybrid");
        assert_eq!(c.tavg, 3);
        assert_eq!(c.prewarp_freeze_at, 3);
        assert_eq!(c.ref_update_conf, 0.3);
    }

    #[test]
    fn standard_is_the_default_config() {
        assert_eq!(
            config_of(&Settings::preset(Preset::Standard)),
            Config::default()
        );
    }

    #[test]
    fn presets_are_recognised_and_edits_are_custom() {
        let mut s = Settings::preset(Preset::Recommended);
        assert_eq!(s.matching_preset(), Some(Preset::Recommended));
        s.tavg = 2;
        assert_eq!(s.matching_preset(), None);
    }

    #[test]
    fn crack_toggle_is_off_by_default_and_not_a_preset() {
        let mut s = Settings::preset(Preset::Recommended);
        assert!(!s.crack);
        assert!(!s.overrides().iter().any(|o| o.starts_with("crack_model")));
        s.crack = true;
        assert_eq!(s.matching_preset(), Some(Preset::Recommended));
        assert_eq!(config_of(&s).crack_model, "xdic");
    }

    #[test]
    fn overrides_round_trip_through_the_cli_parser() {
        let mut s = Settings::preset(Preset::Recommended);
        s.max_disp = 12.5;
        s.sigma_w = 0.1 + 0.2;
        let c = config_of(&s);
        assert_eq!(c.max_disp, 12.5);
        assert_eq!(c.sigma_w, 0.1 + 0.2);
    }
}
