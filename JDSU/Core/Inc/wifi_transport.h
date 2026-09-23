#ifndef __WIFI_TRANSPORT_H
#define __WIFI_TRANSPORT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Fixed 808-byte USB provisioning frame:
 *   0..3   FF FF 01 20
 *   4      version (1)
 *   5      flags: bit0 enable, bit1 persist in STM32 sector 1
 *   6..13  u8 lengths: ssid, wifi password, broker host, client id,
 *           username, MQTT password, topic prefix, device id
 *   14..15 broker port, big endian
 *   16     MQTT scheme: 2 TLS; 1 plaintext is accepted only for testing
 *   17     reserved (must be zero)
 *   18..   the eight fields concatenated in the order above
 *   804..807 IEEE CRC32 over bytes 0..803, big endian
 */
#define WIFI_USB_CONFIG_COMMAND_SIZE 808U

void WifiTransport_Init(void);
void WifiTransport_Process(void);

/* Local USB CDC control has priority over both network paths.  While active,
 * neither LAN TCP nor public MQTT application data/status is sent and network
 * mode commands are ignored.  Releasing CDC restores LAN/MQTT arbitration
 * automatically. */
void WifiTransport_SetLocalControlActive(uint8_t active);

/* IRQ entry points.  They only move bytes/change flags and never parse AT
 * responses, format packets, access flash, or wait for the UART. */
void WifiTransport_OnUartRx(const uint8_t *data, uint16_t length);
void WifiTransport_OnUartTxComplete(void);
void WifiTransport_OnUartError(uint32_t error_code);

/* Returns 1 when frame is the wireless provisioning command (valid or not). */
uint8_t WifiTransport_HandleUsbConfig(const uint8_t *frame, uint16_t length);

/* Queue a read-only 20-byte USB status reply for command FF FF 01 21.
 * The query never changes or persists Wi-Fi credentials. */
uint8_t WifiTransport_HandleUsbStatusQuery(const uint8_t *frame, uint16_t length);

/* usb_scan_frame starts with EE EE/count and contains point-major CH0..CH3
 * 12-bit ADC codes.  The wireless copy owns independent storage, so the scan
 * may immediately reuse the USB buffer. */
void WifiTransport_QueueScanFrame(const uint8_t *usb_scan_frame,
                                  uint16_t point_count,
                                  uint8_t mode,
                                  int32_t temperature_mC,
                                  uint16_t gain20k_mask_ch0,
                                  uint16_t gain20k_mask_ch1,
                                  uint8_t active_channel_mask);

/* LAN raw-control mode mirrors the native USB byte stream.  It is enabled
 * only by an explicit same-LAN TCP client handshake; one seed owns the
 * session until disconnect/timeout, and RAW data never reaches MQTT.
 * Returns 1 when the frame was accepted by the LAN queue. */
uint8_t WifiTransport_QueueRawFrame(const uint8_t *data, uint16_t length);

/* True only while an authenticated same-LAN RAW client owns the control
 * session.  Fast diagnostic streams use this in addition to native USB so a
 * dropped network owner still closes the SOA shutter immediately. */
uint8_t WifiTransport_IsLanRawClientActive(void);

/* Queue one native 453-byte F45 frame in the negotiated LAN-compact format.
 * Only the first direct CH1 sample is transported because that is the
 * qualified realtime position-model input.  The native USB path remains
 * byte-for-byte unchanged. */
uint8_t WifiTransport_QueueFastFullMapFrame(const uint8_t *data,
                                            uint16_t length);

/* Commit a final partial compact-F45 batch before terminal status. */
uint8_t WifiTransport_FlushFastFullMapFrames(void);

/* Reports whether every queued/in-flight LAN RAW frame has completed. */
uint8_t WifiTransport_IsLanRawQueueIdle(void);

/* Queue a native EE/EE scan frame for the LAN RAW stream.  A client may
 * explicitly negotiate a compact on-wire representation; legacy clients and
 * all non-scan frames continue to receive the byte-for-byte USB format. */
uint8_t WifiTransport_QueueRawScanFrame(const uint8_t *data,
                                        uint16_t length,
                                        uint16_t point_count,
                                        uint8_t active_channel_mask);

/* Grants one bounded maintenance window between two complete scan frames.
 * It is used only for the exceptional one-time flash commit requested by a
 * provisioning command; normal AT/MQTT traffic remains fully non-blocking. */
void WifiTransport_NotifyFrameBoundary(void);

/* A command received while scanning is exposed only at a completed-frame
 * boundary.  In idle/manual modes it is exposed at the main-loop safe point. */
uint8_t WifiTransport_TakePendingMode(uint8_t current_mode,
                                      uint8_t at_frame_boundary,
                                      uint8_t *requested_mode,
                                      uint32_t *request_id);
void WifiTransport_OnRemoteModeApplied(uint8_t mode, uint32_t request_id);
void WifiTransport_OnLocalModeChanged(uint8_t mode);

uint8_t WifiTransport_GetState(void);
uint16_t WifiTransport_GetLastError(void);
uint8_t WifiTransport_IsOtaActive(void);
/* Runtime identity shared by native USB and network telemetry.  The table CRC
 * is the CRC32 of the big-endian uint32 wavelength-pm axis for ``mode``. */
uint32_t WifiTransport_GetBootId(void);
uint32_t WifiTransport_GetTableCrc(uint8_t mode);

#ifdef __cplusplus
}
#endif

#endif
