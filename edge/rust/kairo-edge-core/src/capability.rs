//! Device profile + provider capability descriptors and the
//! `select_provider` algorithm.
//!
//! See [`edge/docs/EDGE_ARCHITECTURE.md`](../../docs/EDGE_ARCHITECTURE.md)
//! §5 for the architectural rationale and
//! [`edge/docs/KAIRO_PROTOCOL.md`](../../docs/KAIRO_PROTOCOL.md) §7.3-7.4
//! for the JSON shapes these structs serialize to.

use serde::{Deserialize, Serialize};

/// What kind of provider a manifest describes. Inference, embedding,
/// and storage providers all share the same capability shape but
/// `select_provider` filters by kind so an embedding provider isn't
/// considered for inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    Inference,
    Embedding,
    Storage,
}

/// What a provider declares about itself so the runtime can decide
/// whether to use it on a given device. Capability is read-only at
/// runtime — the provider never lies, the device never overrides.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderCapability {
    pub name: String,
    pub kind: ProviderKind,
    /// Minimum peak RAM in megabytes the provider expects to operate
    /// without thrashing. ``0`` means "no meaningful floor" — the
    /// extractive fallback uses this.
    #[serde(default)]
    pub min_ram_mb: u32,
    #[serde(default)]
    pub needs_gpu: bool,
    #[serde(default)]
    pub needs_webgpu: bool,
    /// Empty list = no model bundled / required. A non-empty list
    /// constrains which model families the provider can load (e.g.
    /// ``["gemma", "llama", "phi"]`` for llama.cpp).
    #[serde(default)]
    pub model_families: Vec<String>,
    /// Free-form version string. Loose-versioned; we use it for
    /// audit logging, not selection.
    pub version: String,
}

impl ProviderCapability {
    pub fn matches(&self, profile: &DeviceProfile) -> bool {
        if self.min_ram_mb > profile.peak_ram_mb {
            return false;
        }
        if self.needs_gpu && !profile.has_gpu {
            return false;
        }
        if self.needs_webgpu && !profile.has_webgpu {
            return false;
        }
        true
    }
}

/// Snapshot of the host the runtime is on. Today this is constructed
/// from named presets (``pi_zero()``, ``pi_4()``, etc.); host probing
/// (sysinfo crate) lands later — kept out of v0.1 to avoid pulling
/// platform-specific dependencies into the edge core.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceProfile {
    pub name: String,
    pub peak_ram_mb: u32,
    pub cpu_cores: u32,
    pub cpu_ghz: f32,
    #[serde(default)]
    pub has_gpu: bool,
    #[serde(default)]
    pub has_webgpu: bool,
    /// Mirrors the Kairo Protocol §7.4 string set: ``linux``,
    /// ``windows``, ``macos``, ``android``, ``ios``, ``browser``.
    #[serde(default = "default_platform")]
    pub platform: String,
    /// ``x86_64`` / ``aarch64`` / ``wasm32`` etc.
    #[serde(default = "default_arch")]
    pub arch: String,
}

fn default_platform() -> String {
    "unknown".to_string()
}

fn default_arch() -> String {
    "unknown".to_string()
}

impl DeviceProfile {
    /// 256 MB, single-core ARMv6 — extractive only.
    pub fn pi_zero() -> Self {
        Self {
            name: "pi-zero".into(),
            peak_ram_mb: 256,
            cpu_cores: 1,
            cpu_ghz: 1.0,
            has_gpu: false,
            has_webgpu: false,
            platform: "linux".into(),
            arch: "armv6".into(),
        }
    }

    /// 1 GB, quad-core Cortex-A72 @ 1.5 GHz. Fits a 1B Q4 LLM.
    pub fn pi_4() -> Self {
        Self {
            name: "pi-4".into(),
            peak_ram_mb: 1024,
            cpu_cores: 4,
            cpu_ghz: 1.5,
            has_gpu: false,
            has_webgpu: false,
            platform: "linux".into(),
            arch: "aarch64".into(),
        }
    }

    /// 2 GB, quad-core Cortex-A76 @ 2.4 GHz. Fits a 3B Q4 LLM.
    pub fn pi_5() -> Self {
        Self {
            name: "pi-5".into(),
            peak_ram_mb: 2048,
            cpu_cores: 4,
            cpu_ghz: 2.4,
            has_gpu: false,
            has_webgpu: false,
            platform: "linux".into(),
            arch: "aarch64".into(),
        }
    }

    /// 1 GB usable, mid-range Android phone class.
    pub fn phone_mid() -> Self {
        Self {
            name: "phone-mid".into(),
            peak_ram_mb: 1024,
            cpu_cores: 4,
            cpu_ghz: 2.0,
            has_gpu: false,
            has_webgpu: false,
            platform: "android".into(),
            arch: "aarch64".into(),
        }
    }

    /// Browser without WebGPU. Pyodide / TS shell on a phone or
    /// low-end laptop. ~400 MB virtual budget.
    pub fn browser_lite() -> Self {
        Self {
            name: "browser-lite".into(),
            peak_ram_mb: 400,
            cpu_cores: 1,
            cpu_ghz: 0.0,
            has_gpu: false,
            has_webgpu: false,
            platform: "browser".into(),
            arch: "wasm32".into(),
        }
    }

    /// Browser with WebGPU. Capable of WebLLM / LiteRT-LM Web.
    pub fn browser_gpu() -> Self {
        Self {
            name: "browser-gpu".into(),
            peak_ram_mb: 1024,
            cpu_cores: 1,
            cpu_ghz: 0.0,
            has_gpu: false,
            has_webgpu: true,
            platform: "browser".into(),
            arch: "wasm32".into(),
        }
    }
}

/// Selection algorithm: from a slice of capabilities filtered by
/// kind, return the one with the highest "richness" that still fits
/// the profile, or `None` if nothing qualifies. Caller decides the
/// fallback (typically: drop to the simplest provider, e.g.
/// extractive for inference).
///
/// Richness ordering today is "highest min_ram_mb that still fits"
/// — capable providers tend to have larger floors. When 14e.4 lands
/// the LlamaCppProvider with `min_ram_mb=800`, it'll outrank
/// `ExtractiveProvider` (`min_ram_mb=0`) on pi-4 (1024 MB), and on
/// pi-zero (256 MB) the LLM gets filtered out so extractive wins.
pub fn select_capability<'a>(
    candidates: &'a [&'a ProviderCapability],
    profile: &DeviceProfile,
    kind: ProviderKind,
) -> Option<&'a ProviderCapability> {
    candidates
        .iter()
        .filter(|c| c.kind == kind)
        .filter(|c| c.matches(profile))
        .copied()
        .max_by_key(|c| c.min_ram_mb)
}


#[cfg(test)]
mod tests {
    use super::*;

    fn extractive_cap() -> ProviderCapability {
        ProviderCapability {
            name: "extractive".into(),
            kind: ProviderKind::Inference,
            min_ram_mb: 0,
            needs_gpu: false,
            needs_webgpu: false,
            model_families: vec![],
            version: "0.1.0".into(),
        }
    }

    fn llama_cpp_cap() -> ProviderCapability {
        ProviderCapability {
            name: "llama-cpp".into(),
            kind: ProviderKind::Inference,
            min_ram_mb: 800,
            needs_gpu: false,
            needs_webgpu: false,
            model_families: vec!["llama".into(), "gemma".into()],
            version: "0.1.0".into(),
        }
    }

    fn webgpu_cap() -> ProviderCapability {
        ProviderCapability {
            name: "webllm".into(),
            kind: ProviderKind::Inference,
            min_ram_mb: 800,
            needs_gpu: false,
            needs_webgpu: true,
            model_families: vec!["llama".into()],
            version: "0.1.0".into(),
        }
    }

    #[test]
    fn pi_zero_picks_extractive_only() {
        let ext = extractive_cap();
        let llama = llama_cpp_cap();
        let candidates = [&ext, &llama];
        let chosen = select_capability(
            &candidates, &DeviceProfile::pi_zero(), ProviderKind::Inference,
        );
        assert_eq!(chosen.unwrap().name, "extractive");
    }

    #[test]
    fn pi_4_picks_llama_cpp_over_extractive() {
        let ext = extractive_cap();
        let llama = llama_cpp_cap();
        let candidates = [&ext, &llama];
        let chosen = select_capability(
            &candidates, &DeviceProfile::pi_4(), ProviderKind::Inference,
        );
        assert_eq!(chosen.unwrap().name, "llama-cpp");
    }

    #[test]
    fn webgpu_only_provider_skipped_without_webgpu() {
        let webllm = webgpu_cap();
        let candidates = [&webllm];
        // browser-lite has no WebGPU
        let chosen = select_capability(
            &candidates, &DeviceProfile::browser_lite(), ProviderKind::Inference,
        );
        assert!(chosen.is_none());
    }

    #[test]
    fn webgpu_only_provider_chosen_on_browser_gpu() {
        let webllm = webgpu_cap();
        let candidates = [&webllm];
        let chosen = select_capability(
            &candidates, &DeviceProfile::browser_gpu(), ProviderKind::Inference,
        );
        assert_eq!(chosen.unwrap().name, "webllm");
    }

    #[test]
    fn empty_candidates_returns_none() {
        let chosen = select_capability(&[], &DeviceProfile::pi_4(), ProviderKind::Inference);
        assert!(chosen.is_none());
    }

    #[test]
    fn kind_filter_skips_wrong_kind_providers() {
        let mut emb = extractive_cap();
        emb.kind = ProviderKind::Embedding;
        emb.name = "hash-stub".into();
        let candidates = [&emb];
        let chosen = select_capability(
            &candidates, &DeviceProfile::pi_4(), ProviderKind::Inference,
        );
        assert!(chosen.is_none());
    }

    #[test]
    fn capability_matches_obeys_ram_floor() {
        let mut cap = llama_cpp_cap();
        cap.min_ram_mb = 4096;
        assert!(!cap.matches(&DeviceProfile::pi_4()));
        assert!(cap.matches(&DeviceProfile::pi_5()) == false); // 2 GB < 4 GB
    }

    #[test]
    fn capability_roundtrips_through_json() {
        let cap = llama_cpp_cap();
        let json = serde_json::to_string(&cap).unwrap();
        let back: ProviderCapability = serde_json::from_str(&json).unwrap();
        assert_eq!(cap.name, back.name);
        assert_eq!(cap.min_ram_mb, back.min_ram_mb);
        assert_eq!(cap.model_families, back.model_families);
    }

    #[test]
    fn device_profile_roundtrips_through_json() {
        let profile = DeviceProfile::phone_mid();
        let json = serde_json::to_string(&profile).unwrap();
        let back: DeviceProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(profile.name, back.name);
        assert_eq!(profile.platform, back.platform);
        assert_eq!(profile.arch, back.arch);
    }
}
