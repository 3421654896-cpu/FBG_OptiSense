#ifndef __OTA_UPDATE_H
#define __OTA_UPDATE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void OtaUpdate_Init(uint32_t runtime_boot_id);

/* Returns one when payload contains a complete FOTA1 command.  The response
 * is newline-terminated so a TCP client can recover it from any stale binary
 * spectrum bytes already buffered by the radio module. */
uint8_t OtaUpdate_HandleCommand(const char *payload,
                                uint32_t payload_length,
                                const char *authorization_key,
                                char *response,
                                uint16_t response_capacity,
                                uint8_t *reboot_after_response);

uint8_t OtaUpdate_IsActive(void);
uint32_t OtaUpdate_ReceivedBytes(void);
void OtaUpdate_OnClientDisconnected(void);
/* Release only the volatile transfer lock after a fully verified image has
 * already been committed to metadata but the coordinated radio/MCU restart
 * could not be confirmed.  The bootloader-visible pending image is preserved
 * for the next safe reset. */
void OtaUpdate_ReleaseCommittedSession(void);

#ifdef __cplusplus
}
#endif

#endif
