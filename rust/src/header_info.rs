use anyhow::{bail, Context};
use serde::Deserialize;

/// Describes a thermal camera's specs sent at connection time.
/// Mirrors the Python `HeaderInfo` class in mediumpower.py.
#[derive(Debug, Clone)]
pub struct HeaderInfo {
    pub medium_power: bool,
    pub res_x: u32,
    pub res_y: u32,
    pub fps: u32,
    pub brand: String,
    pub model: String,
    pub frame_size: u32,
    pub pixel_bits: u32,
    pub serial: String,
    pub firmware: String,
}

impl Default for HeaderInfo {
    fn default() -> Self {
        Self {
            medium_power: false,
            res_x: 160,
            res_y: 120,
            fps: 9,
            brand: "lepton".to_string(),
            model: "lepton3.5".to_string(),
            frame_size: 39040,
            pixel_bits: 16,
            serial: "12".to_string(),
            firmware: "12".to_string(),
        }
    }
}

/// Raw YAML structure used during parsing.
#[derive(Deserialize, Debug)]
struct RawHeader {
    #[serde(alias = "ResX")]   res_x:      Option<u32>,
    #[serde(alias = "ResY")]   res_y:      Option<u32>,
    #[serde(alias = "FPS")]    fps:        Option<u32>,
    #[serde(alias = "Model")]  model:      Option<String>,
    #[serde(alias = "Brand")]  brand:      Option<String>,
    #[serde(alias = "PixelBits")] pixel_bits: Option<u32>,
    #[serde(alias = "FrameSize")] frame_size: Option<u32>,
    #[serde(alias = "CameraSerial")] serial: Option<String>,
    #[serde(alias = "Firmware")] firmware:  Option<serde_yaml::Value>,
}

impl HeaderInfo {
    /// Parse a raw header string received from the camera connection.
    /// The special string `"medium"` signals medium-power streaming mode.
    pub fn parse_header(raw: &str) -> anyhow::Result<Self> {
        if raw.trim() == "medium" {
            return Ok(Self { medium_power: true, ..Default::default() });
        }

        let raw: RawHeader =
            serde_yaml::from_str(raw).context("parsing camera header YAML")?;

        let firmware = raw
            .firmware
            .as_ref()
            .map(|v| match v {
                serde_yaml::Value::String(s) => s.clone(),
                other => format!("{:?}", other),
            })
            .unwrap_or_default();

        let mut h = Self {
            medium_power: false,
            res_x:      raw.res_x.unwrap_or(160),
            res_y:      raw.res_y.unwrap_or(120),
            fps:        raw.fps.unwrap_or(9),
            brand:      raw.brand.unwrap_or_else(|| "lepton".to_string()),
            model:      raw.model.unwrap_or_else(|| "lepton3.5".to_string()),
            frame_size: raw.frame_size.unwrap_or(0),
            pixel_bits: raw.pixel_bits.unwrap_or(0),
            serial:     raw.serial.unwrap_or_else(|| "".to_string()),
            firmware,
        };

        // Derive missing field from the other.
        if h.res_x > 0 && h.res_y > 0 {
            if h.pixel_bits == 0 && h.frame_size > 0 {
                h.pixel_bits = 8 * h.frame_size / (h.res_x * h.res_y);
            } else if h.frame_size == 0 && h.pixel_bits > 0 {
                h.frame_size = h.res_x * h.res_y * h.pixel_bits / 8;
            }
        }

        h.validate()?;
        Ok(h)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        if self.res_x == 0 || self.res_y == 0 || self.fps == 0 || self.pixel_bits == 0 {
            bail!(
                "header info is missing a required field (ResX, ResY, FPS and/or PixelBits)"
            );
        }
        Ok(())
    }
}

impl std::fmt::Display for HeaderInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HeaderInfo {{ res={}x{} fps={} model={} frame_size={} pixel_bits={} }}",
            self.res_x, self.res_y, self.fps, self.model, self.frame_size, self.pixel_bits
        )
    }
}
