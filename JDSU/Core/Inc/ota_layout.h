#ifndef __OTA_LAYOUT_H
#define __OTA_LAYOUT_H

#include <stdint.h>

/* STM32F205RE 512-KB single-bank flash layout.
 *
 *   sector 0       0x08000000..0x08003FFF  immutable OTA bootloader
 *   sector 1       0x08004000..0x08007FFF  Wi-Fi config + OTA metadata
 *   sectors 2..5   0x08008000..0x0803FFFF  active application (224 KB)
 *   sectors 6..7   0x08040000..0x0807FFFF  received image (256 KB)
 */
#define OTA_BOOT_ADDRESS              0x08000000UL
#define OTA_CONFIG_ADDRESS            0x08004000UL
#define OTA_METADATA_ADDRESS          0x08006000UL
#define OTA_APPLICATION_ADDRESS       0x08008000UL
#define OTA_APPLICATION_END           0x08040000UL
#define OTA_STAGING_ADDRESS           0x08040000UL
#define OTA_STAGING_END               0x08080000UL
#define OTA_APPLICATION_MAX_SIZE      (OTA_APPLICATION_END - OTA_APPLICATION_ADDRESS)
#define OTA_STAGING_MAX_SIZE          (OTA_STAGING_END - OTA_STAGING_ADDRESS)
#define OTA_CONFIG_PRESERVE_SIZE      1024U

#define OTA_TARGET_ID                 0xF2050052UL
#define OTA_METADATA_MAGIC_PENDING    0x4F544131UL /* OTA1 */
#define OTA_METADATA_FORMAT_VERSION   1UL

typedef struct
{
    uint32_t magic;
    uint32_t format_version;
    uint32_t target_id;
    uint32_t image_version;
    uint32_t image_size;
    uint32_t image_crc32;
    uint32_t link_address;
    uint32_t staging_address;
    /* reserved[0] is the install boot token.  reserved[6] is committed erased
     * and is available to the bootloader as a one-way, single-retry marker.
     * The bootloader clears only the magic word after a successful copy, so
     * the newly started application can prove to the OTA client that it is a
     * different boot instance. */
    uint32_t reserved[7];
    uint32_t metadata_crc32;
} OtaMetadata;

#endif
