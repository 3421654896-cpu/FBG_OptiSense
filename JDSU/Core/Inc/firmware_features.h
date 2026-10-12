#ifndef FBG_FIRMWARE_FEATURES_H
#define FBG_FIRMWARE_FEATURES_H

/* Customer firmware omits independent features hidden in the desktop UI.
 * A developer build may override these with compiler definitions. The full
 * pre-trim source and binaries are also preserved in backups/. */
#ifndef FBG_FEATURE_LEGACY_PEAK_MODES
#define FBG_FEATURE_LEGACY_PEAK_MODES 0
#endif
#ifndef FBG_FEATURE_MQTT
#define FBG_FEATURE_MQTT 0
#endif
/* Removed from active source, not a recoverable build-time option. */
#if defined(FBG_FEATURE_NETWORK_OTA) && FBG_FEATURE_NETWORK_OTA
#error Network firmware updates have been removed; use physical USB
#endif
#define FBG_FEATURE_NETWORK_OTA 0

/* Developer-only opt-in. Never enabled in the formal/customer target. */
#ifndef FBG_FEATURE_BETA_SINGLE_ADC
#define FBG_FEATURE_BETA_SINGLE_ADC 0
#endif

#ifndef FBG_FEATURE_BETA_DAC_POLLING
#define FBG_FEATURE_BETA_DAC_POLLING 0
#endif
#if FBG_FEATURE_BETA_DAC_POLLING && !FBG_FEATURE_BETA_SINGLE_ADC
#error DAC polling is only available in the explicitly selected Beta target
#endif

/* USB firmware update, LAN RAW transport, dense/temporary point acquisition,
 * direct DAC calibration, thermal control and fail-dark protections are shared
 * by visible workflows and must not be excluded with the hidden features. */
#endif
